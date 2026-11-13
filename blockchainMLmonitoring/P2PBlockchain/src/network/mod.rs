// src/network/mod.rs (исправлен импорт futures)
use async_trait::async_trait;
use futures::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt}; // Теперь futures в зависимостях
use libp2p::request_response::{self};
use libp2p::swarm::NetworkBehaviour;
use libp2p::gossipsub;

#[derive(Clone)]
pub struct AppProtocol;

impl AsRef<str> for AppProtocol {
    fn as_ref(&self) -> &str { "/app/1" }
}

#[derive(Clone, Default)]
pub struct AppCodec;

#[async_trait]
impl request_response::Codec for AppCodec {
    type Protocol = AppProtocol;
    type Request = String;
    type Response = String;

    async fn read_request<T>(&mut self, _: &Self::Protocol, io: &mut T) -> std::io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let mut buf = Vec::new();
        io.read_to_end(&mut buf).await?;
        Ok(String::from_utf8_lossy(&buf).to_string())
    }

    async fn read_response<T>(&mut self, _: &Self::Protocol, io: &mut T) -> std::io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let mut buf = Vec::new();
        io.read_to_end(&mut buf).await?;
        Ok(String::from_utf8_lossy(&buf).to_string())
    }

    async fn write_request<T>(&mut self, _: &Self::Protocol, io: &mut T, req: Self::Request) -> std::io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        io.write_all(req.as_bytes()).await
    }

    async fn write_response<T>(&mut self, _: &Self::Protocol, io: &mut T, res: Self::Response) -> std::io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        io.write_all(res.as_bytes()).await
    }
}

#[derive(NetworkBehaviour)]
pub struct AppBehaviour {
    pub rr: request_response::Behaviour<AppCodec>,
    pub gossipsub: gossipsub::Behaviour,
}