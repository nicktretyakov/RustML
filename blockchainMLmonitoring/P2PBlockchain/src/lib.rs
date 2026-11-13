// src/lib.rs (добавлены новые модули)
pub mod blockchain;
pub mod market;
pub mod network;
pub mod messaging;
pub mod tasks;
pub mod iot;
pub mod storage;

pub use blockchain::BlockchainService;
pub use market::{MarketService, Product};
pub use messaging::ServerState;
pub use tasks::{Task, Node};
pub use iot::IoTNetwork;
pub use storage::DecentralizedStorage;