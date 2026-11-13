// src/main.rs
// Полная версия с параллельным REST API (Axum на порту 8081)
// iot_network: Arc<Mutex<IoTNetwork>> для mutable доступа
// let mut state для mutable borrow
// Все ошибки исправлены (map с &data для from_utf8_lossy)
// P2P swarm + REST API параллельно
// Готово к ML-валидации в api.rs (ансамбль + LOF на GPU через contextML)

mod api;
mod blockchain;
mod iot;
mod market;
mod messaging;
mod network;
mod storage;
mod tasks;

use futures::stream::StreamExt;
use lazy_static::lazy_static;
use libp2p::gossipsub::{IdentTopic, MessageAuthenticity};
use libp2p::gossipsub::Event as GossipsubEvent;
use libp2p::identity;
use libp2p::request_response;
use libp2p::request_response::Event as RrEvent;
use libp2p::request_response::Message as RrMessage;
use libp2p::SwarmBuilder;
use libp2p::swarm::{SwarmEvent, Config as SwarmConfig};
use libp2p::{noise, tcp, yamux};
use libp2p::{Multiaddr, PeerId};
use serde_json;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::signal;
use tokio::sync::mpsc;

use crate::blockchain::BlockchainService;
use crate::iot::IoTNetwork;
use crate::market::{MarketService, Product};
use crate::messaging::{Message as ChatMessage, ServerState};
use crate::network::{AppBehaviour, AppCodec, AppProtocol, AppBehaviourEvent};
use crate::storage::DecentralizedStorage;
use crate::tasks::{Node, Task};

lazy_static! {
    static ref APP_STATE: Mutex<AppState> = Mutex::new(AppState::new());
}

struct AppState {
    blockchain: Arc<Mutex<BlockchainService>>,
    market: Arc<Mutex<MarketService>>,
    server_state: Arc<Mutex<ServerState>>,
    iot_network: Arc<Mutex<IoTNetwork>>,
    storage: Arc<Mutex<DecentralizedStorage>>,
    user_files: Arc<Mutex<std::collections::HashMap<String, Vec<u8>>>>,
    task_sender: mpsc::Sender<Task>,
    nodes: Vec<Arc<Node>>,
    task_receiver: Option<mpsc::Receiver<Task>>,
}

impl AppState {
    fn new() -> Self {
        let (task_sender, task_receiver) = mpsc::channel(100);
        Self {
            blockchain: Arc::new(Mutex::new(BlockchainService::new())),
            market: Arc::new(Mutex::new(MarketService::new())),
            server_state: Arc::new(Mutex::new(ServerState::new())),
            iot_network: Arc::new(Mutex::new(IoTNetwork::new())),
            storage: Arc::new(Mutex::new(DecentralizedStorage::new())),
            user_files: Arc::new(Mutex::new(std::collections::HashMap::new())),
            task_sender,
            nodes: vec![Arc::new(Node::new(1)), Arc::new(Node::new(2))],
            task_receiver: Some(task_receiver),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let local_key = identity::Keypair::generate_ed25519();
    let local_peer_id = PeerId::from(local_key.public());
    println!("Local peer ID: {local_peer_id}");

    // Инициализация состояния
    {
        let mut state = APP_STATE.lock().unwrap();

        {
            let mut files = state.user_files.lock().unwrap();
            files.insert("file1.txt".to_string(), b"Content of file 1".to_vec());
            files.insert("file2.txt".to_string(), b"Content of file 2".to_vec());
        }

        let mut receiver = state.task_receiver.take().expect("Receiver already taken");
        let nodes = state.nodes.clone();

        tokio::spawn(async move {
            while let Some(task) = receiver.recv().await {
                for node in &nodes {
                    node.add_task(task.clone());
                    node.process_tasks().await;
                }
            }
        });
    }

    // Behaviour
    let behaviour = {
        let mut gossipsub = libp2p::gossipsub::Behaviour::new(
            MessageAuthenticity::Signed(local_key.clone()),
            libp2p::gossipsub::Config::default(),
        )?;
        gossipsub.subscribe(&IdentTopic::new("global"))?;

        let rr = request_response::Behaviour::<AppCodec>::new(
            [(AppProtocol, request_response::ProtocolSupport::Full)],
            request_response::Config::default(),
        );

        AppBehaviour { rr, gossipsub }
    };

    // Swarm
    let mut swarm = SwarmBuilder::with_existing_identity(local_key.clone())
        .with_tokio()
        .with_tcp(tcp::Config::default(), noise::Config::new, || yamux::Config::default())?
        .with_behaviour(|_| behaviour)?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

    if let Some(addr_str) = std::env::args().nth(1) {
        let remote_addr: Multiaddr = addr_str.parse()?;
        swarm.dial(remote_addr.clone())?;
        println!("Dialed {remote_addr}");
    }

    // Параллельный REST API
    let api_router = api::create_api_router();
    let _api_task = tokio::spawn(async move {
        let listener = TcpListener::bind("0.0.0.0:8081").await.unwrap();
        println!("🚀 REST API запущен на http://0.0.0.0:8081");
        axum::serve(listener, api_router).await.unwrap();
    });

    // Swarm loop
    loop {
        tokio::select! {
            event = swarm.select_next_some() => {
                match event {
                    SwarmEvent::NewListenAddr { address, .. } => println!("Listening on {address}"),
                    SwarmEvent::Behaviour(AppBehaviourEvent::Rr(event)) => {
                        if let RrEvent::Message { message, .. } = event {
                            match message {
                                RrMessage::Request { request, channel, .. } => {
                                    let response = process_request(request).await;
                                    swarm.behaviour_mut().rr.send_response(channel, response).ok();
                                }
                                RrMessage::Response { response, .. } => {
                                    println!("Received response: {response}");
                                }
                            }
                        }
                    }
                    SwarmEvent::Behaviour(AppBehaviourEvent::Gossipsub(GossipsubEvent::Message { message, .. })) => {
                        let text = String::from_utf8_lossy(&message.data);
                        println!("Gossip received: {text}");
                    }
                    _ => {}
                }
            }
            _ = signal::ctrl_c() => {
                println!("\n📶 Ctrl+C — завершение");
                break;
            }
        }
    }

    Ok(())
}

// Полный process_request (исправлен map с &data)
async fn process_request(request: String) -> String {
    let mut state = APP_STATE.lock().unwrap();

    let parts: Vec<&str> = request.split(',').collect();

    if parts.is_empty() {
        return "Invalid request".to_string();
    }

    match parts[0] {
        "GET_FILE" if parts.len() == 2 => state.user_files.lock().unwrap().get(parts[1])
            .map(|data| String::from_utf8_lossy(&data).to_string())
            .unwrap_or("File not found".to_string()),

        "SEND_AUDIO" => "Dummy audio content".to_string(),

        "ADD_BLOCK" if parts.len() >= 2 => {
            let data = parts[1..].join(",");
            state.blockchain.lock().unwrap().add_block(data);
            "Block added".to_string()
        }

        "SEND_MESSAGE" if parts.len() >= 3 => {
            let sender = parts[1].to_string();
            let content = parts[2..].join(",");
            let message = ChatMessage { sender, content };
            state.server_state.lock().unwrap().broadcast(message);
            "Message sent".to_string()
        }

        "ADD_TASK" if parts.len() >= 2 => {
            let data = parts[1..].join(",").into_bytes();
            let task = Task { data };
            let sender = state.task_sender.clone();
            if sender.send(task).await.is_ok() {
                "Task added".to_string()
            } else {
                "Task send failed".to_string()
            }
        }

        "UPDATE_IOT" if parts.len() == 3 => {
            let device_id = parts[1];
            let new_data = parts[2];
            state.iot_network.lock().unwrap().add_device(device_id);
            state.iot_network.lock().unwrap().update_device_data(device_id, new_data);
            "IoT updated".to_string()
        }

        "STORE" if parts.len() == 3 => {
            let key = parts[1].to_string();
            let value = parts[2].as_bytes().to_vec();
            state.storage.lock().unwrap().store_data(key, value);
            "Data stored".to_string()
        }

        "RETRIEVE" if parts.len() == 2 => state.storage.lock().unwrap().retrieve_data(parts[1])
            .map(|data| String::from_utf8_lossy(&data).to_string())
            .unwrap_or("Key not found".to_string()),

        "ADD_PRODUCT" if parts.len() == 3 => {
            let name = parts[1].to_string();
            let price: f64 = parts[2].parse().unwrap_or(0.0);
            state.market.lock().unwrap().add_product(name.clone(), price);
            state.blockchain.lock().unwrap().add_block(format!("Added product {name} at price {price}"));
            "Product added".to_string()
        }

        "GET_PRODUCTS" => {
            let products = state.market.lock().unwrap().get_products();
            serde_json::to_string(&products).unwrap_or("JSON error".to_string())
        }

        "PREDICT_PRICE" if parts.len() == 2 => {
            let name = parts[1];
            let pred = state.market.lock().unwrap().predict_next_price(name);
            format!("Predicted price: {:.2}", pred)
        }

        _ => "Invalid command".to_string(),
    }
}