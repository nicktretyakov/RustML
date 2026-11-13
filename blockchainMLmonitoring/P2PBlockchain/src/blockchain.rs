use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use chrono::Utc;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Block {
    pub index: u64,
    pub timestamp: i64, // Unix timestamp
    pub data: String,
    pub previous_hash: String,
    pub nonce: u64, // Для PoW в будущем (0 по умолчанию)
    pub hash: String,
}

impl Block {
    pub fn new(index: u64, timestamp: i64, data: String, previous_hash: String) -> Self {
        let mut block = Block {
            index,
            timestamp,
            data,
            previous_hash,
            nonce: 0,
            hash: String::new(),
        };
        block.hash = block.calculate_hash();
        block
    }

    pub fn calculate_hash(&self) -> String {
        let mut hasher = Sha256::new();
        // 5 полей → 5 плейсхолдеров (index, timestamp, data, previous_hash, nonce)
        hasher.update(format!("{}{}{}{}{}", self.index, self.timestamp, self.data, self.previous_hash, self.nonce));
        format!("{:x}", hasher.finalize())
    }
}

pub struct BlockchainService {
    chain: Vec<Block>,
}

impl BlockchainService {
    pub fn new() -> Self {
        let genesis = Block::new(0, Utc::now().timestamp(), "Genesis Block".to_string(), "0".to_string());
        Self { chain: vec![genesis] }
    }

    pub fn add_block(&mut self, data: String) {
        let previous = self.chain.last().unwrap().clone();
        let new_block = Block::new(
            previous.index + 1,
            Utc::now().timestamp(),
            data,
            previous.hash,
        );
        self.chain.push(new_block);
    }

    pub fn get_chain(&self) -> &Vec<Block> {
        &self.chain
    }
}