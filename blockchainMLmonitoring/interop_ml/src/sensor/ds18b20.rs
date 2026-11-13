use std::{fs::File, io::{BufRead, BufReader}, path::Path};
use super::{TemperatureSensor, SensorError};

pub struct Ds18b20Sensor {
    device_path: String,
    retries: usize,
}

impl Ds18b20Sensor {
    pub fn new(device_id: &str, retries: usize) -> Self {
        Self {
            device_path: format!("/sys/bus/w1/devices/{}/w1_slave", device_id),
            retries,
        }
    }

    fn read_raw(&self) -> Result<f64, SensorError> {
        let path = Path::new(&self.device_path);
        if !path.exists() {
            return Err(SensorError::DeviceNotFound);
        }

        let file = File::open(path).map_err(SensorError::IoError)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let first_line = lines.next()
            .ok_or(SensorError::ParseError("Empty file".into()))?
            .map_err(SensorError::IoError)?;

        let second_line = lines.next()
            .ok_or(SensorError::ParseError("Missing second line".into()))?
            .map_err(SensorError::IoError)?;

        if !first_line.ends_with("YES") {
            return Err(SensorError::CrcError);
        }

        let temp_str = second_line
            .split("t=")
            .nth(1)
            .ok_or(SensorError::ParseError("No t= field".into()))?;

        let millis: i32 = temp_str.trim().parse()
            .map_err(|_| SensorError::ParseError("Invalid temperature value".into()))?;

        Ok(millis as f64 / 1000.0)
    }
}

impl TemperatureSensor for Ds18b20Sensor {
    fn read(&self) -> Result<f64, SensorError> {
        for _ in 0..self.retries {
            match self.read_raw() {
                Ok(temp) => return Ok(temp),
                Err(SensorError::CrcError) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(SensorError::Timeout)
    }
}