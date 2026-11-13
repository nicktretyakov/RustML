// src/main.rs

#[macro_use]
extern crate rocket;

mod data_loader;
mod preprocessor;
mod pipeline;
mod ui_components;

use data_loader::load_raw_data;
use preprocessor::preprocess;
use pipeline::{Pipeline, GradientDescentLinearModel, OptimizerType, NoiseAugmentation, FittedModel};
use std::sync::Arc;

use ui_components::{
    DynComponent, UIDescription, generate_ui,
};

use actix_web::{App, HttpResponse, HttpServer, Responder, web};
use anyhow::Result;
use futures::{SinkExt, StreamExt};
use reqwest::Client;
use rocket_dyn_templates::{Template, context};
use std::collections::HashMap;
use std::thread;
use tokio::signal;
use warp::Filter;
use warp::ws::{Message, WebSocket};
use ndarray::{Array2, ArrayView2};

// --- Rocket ---
#[get("/")]
fn index() -> Template {
    Template::render(
        "index",
        context! {
            message: "Hello, from the backend!",
        },
    )
}

async fn rocket_launch() -> Result<(), rocket::Error> {
    println!("🚀 Starting Rocket on http://127.0.0.1:8000");
    let rocket = rocket::build()
        .attach(Template::fairing())
        .mount("/", routes![index]);
    rocket.launch().await?;
    Ok(())
}

// --- Actix Web API ---
async fn api_handler() -> impl Responder {
    use ui_components::{DeserUIElement, UserData, UIDescription};

    let elements = vec![
        DeserUIElement::Text {
            content: "Привет, мир!".to_string(),
            style: None,
            on_click: None,
        },
        DeserUIElement::Button {
            label: "Нажми на меня".to_string(),
            action: "submit".to_string(),
            style: None,
            on_click: None,
        },
    ];
    let user_data = UserData {
        age: 25,
        preferences: vec!["dark_mode".to_string(), "large_text".to_string()],
    };
    let ui_desc = UIDescription { elements, user_data };
    HttpResponse::Ok().json(ui_desc)
}

async fn actix_main() -> std::io::Result<()> {
    println!("🌐 Starting Actix Web on http://127.0.0.1:8081/api");
    HttpServer::new(|| App::new().service(web::resource("/api").to(api_handler)))
        .bind("127.0.0.1:8081")?
        .run()
        .await
}

// --- Warp WebSocket ---
async fn handle_websocket(mut websocket: WebSocket) {
    while let Some(result) = websocket.next().await {
        let msg = match result {
            Err(e) => {
                eprintln!("WebSocket error: {}", e);
                break;
            }
            Ok(msg) => msg,
        };
        if let Ok(text) = msg.to_str() {
            println!("Received: {}", text);
            if websocket.send(Message::text(text.to_string())).await.is_err() {
                break;
            }
        }
    }
}

async fn warp_main() {
    println!("🔗 Starting Warp WebSocket on ws://127. tender:0.0.1:3030/ws");
    let ws_route = warp::path!("ws")
        .and(warp::ws())
        .map(|ws: warp::ws::Ws| ws.on_upgrade(handle_websocket));
    warp::serve(ws_route).run(([127, 0, 0, 1], 3030)).await;
}

// --- Fetch UI ---
async fn fetch_ui_description(client: &Client, url: &str) -> Result<UIDescription> {
    let response = client.get(url).send().await?;
    let ui_desc: UIDescription = response.json().await?;
    Ok(ui_desc)
}

// --- predict_element_order ---
fn predict_element_order(
    fitted: &dyn FittedModel,
    age: f64,
    has_large_text: bool,
    num_elements: usize,
) -> Vec<usize> {
    let pref_val = if has_large_text { 1.0 } else { 0.0 };
    let features_vec = vec![age, pref_val];
    let features = Array2::from_shape_vec((1, 2), features_vec).unwrap();

    let prediction = fitted.predict(features.view());
    let first_index = prediction[0].round().abs() as usize % num_elements;

    let mut order: Vec<usize> = (0..num_elements).collect();
    order.remove(first_index);
    order.insert(0, first_index);
    order
}

// --- process_ui ---
async fn process_ui(
    client: &Client,
    url: &str,
    fitted: &dyn FittedModel,
    cache: &mut HashMap<String, Vec<usize>>,
) -> Result<()> {
    let ui_desc = fetch_ui_description(client, url).await?;

    let mut dyn_elements: Vec<Box<DynComponent>> = 
        ui_desc.elements.into_iter().map(|e| e.to_component()).collect();

    for element in &dyn_elements {
        element.validate()?;
    }

    let user_id = "user123";
    let has_large_text = ui_desc.user_data.preferences.contains(&"large_text".to_string());

    let order = if let Some(cached_order) = cache.get(user_id) {
        cached_order.clone()
    } else {
        let predicted_order = predict_element_order(
            fitted,
            ui_desc.user_data.age as f64,
            has_large_text,
            dyn_elements.len(),
        );
        cache.insert(user_id.to_string(), predicted_order.clone());
        predicted_order
    };

    generate_ui(&dyn_elements, &order, 0)?;
    Ok(())
}

// --- main ---
#[tokio::main]
async fn main() -> Result<()> {
    let _rocket_handle = thread::spawn(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(rocket_launch()).unwrap();
    });

    let _actix_handle = thread::spawn(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(actix_main()).unwrap();
    });

    let _warp_handle = thread::spawn(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(warp_main());
    });

    let client = Client::new();
    let url = "http://127.0.0.1:8081/api";

    // === Пайплайн с градиентным спуском (Adam) и параллельной аугментацией ===
    let (records, targets) = load_raw_data();
    let raw_dataset = preprocess(records, targets);

    let gd_model = GradientDescentLinearModel {
        learning_rate: 0.01,
        epochs: 1000,
        batch_size: 32,
        add_bias: true,
        optimizer: OptimizerType::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
    };

    let pipeline = Pipeline::new(Arc::new(gd_model))
        .add_transformer(Arc::new(NoiseAugmentation { std_dev: 0.01 }));

    let fitted = pipeline.fit(&raw_dataset)?;

    let mut cache: HashMap<String, Vec<usize>> = HashMap::new();

    if let Err(e) = process_ui(&client, url, fitted.as_ref(), &mut cache).await {
        eprintln!("Warning: UI processing failed (Actix may still be starting): {}", e);
    }

    println!("\n=== Все серверы запущены ===");
    println!("🚀 Rocket:     http://127.0.0.1:8000");
    println!("🌐 Actix API:  http://127.0.0.1:8081/api");
    println!("🔗 Warp WS:    ws://127.0.0.1:3030/ws");
    println!("Нажмите Ctrl+C для остановки\n");

    signal::ctrl_c().await.expect("Failed to listen for Ctrl+C");
    println!("Получен сигнал завершения. Останавливаем...");

    Ok(())
}