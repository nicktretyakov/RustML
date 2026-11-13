// src/ui_components.rs

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

// --- Отдельные интерфейсы (трейты) вместо одного большого ---
// Принцип Interface Segregation: клиенты (код, использующий компоненты) не зависят от методов, которые им не нужны.

// 1. Интерфейс для валидации (используется только при проверке структуры UI)
pub trait Validatable: Send + Sync + Debug {
    fn validate(&self) -> Result<()>;
}

// 2. Интерфейс для рендеринга (используется только при выводе UI в консоль)
pub trait Renderable: Send + Sync + Debug {
    fn render(&self, indent: usize) -> Result<()>;
}

// --- Тип для полиморфных компонентов (комбинируем нужные трейты) ---
// Теперь любой компонент должен реализовывать оба интерфейса (для текущей логики),
// но если в будущем появится компонент, которому нужна только валидация или только рендеринг — он реализует только нужный трейт.
pub trait UIComponent: Validatable + Renderable {}
impl<T: Validatable + Renderable> UIComponent for T {}

pub type DynComponent = dyn UIComponent;

// --- Конкретные компоненты (реализуют нужные трейты отдельно) ---
#[derive(Debug, Clone)]
pub struct TextElement {
    pub content: String,
    pub style: Option<String>,
    pub on_click: Option<String>,
}

impl Validatable for TextElement {
    fn validate(&self) -> Result<()> {
        if self.content.is_empty() {
            Err(anyhow::anyhow!("Элемент 'text' должен содержать непустой 'content'"))
        } else {
            Ok(())
        }
    }
}

impl Renderable for TextElement {
    fn render(&self, indent: usize) -> Result<()> {
        let indent_str = " ".repeat(indent);
        let style = self.style.as_ref().map_or("".to_string(), |s| format!(" [стиль: {}]", s));
        let on_click = self.on_click.as_ref().map_or("".to_string(), |o| format!(" [on_click: {}]", o));
        println!("{}{}Текст: {}{}{}", indent_str, "text", self.content, style, on_click);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ButtonElement {
    pub label: String,
    pub action: String,
    pub style: Option<String>,
    pub on_click: Option<String>,
}

impl Validatable for ButtonElement {
    fn validate(&self) -> Result<()> {
        if self.label.is_empty() || self.action.is_empty() {
            Err(anyhow::anyhow!("Элемент 'button' должен содержать непустые 'label' и 'action'"))
        } else {
            Ok(())
        }
    }
}

impl Renderable for ButtonElement {
    fn render(&self, indent: usize) -> Result<()> {
        let indent_str = " ".repeat(indent);
        let style = self.style.as_ref().map_or("".to_string(), |s| format!(" [стиль: {}]", s));
        let on_click = self.on_click.as_ref().map_or("".to_string(), |o| format!(" [on_click: {}]", o));
        println!(
            "{}{}: {} (действие: {}{}{})",
            indent_str, "button", self.label, self.action, style, on_click
        );
        Ok(())
    }
}

#[derive(Debug)]
pub struct ContainerElement {
    pub children: Vec<Box<DynComponent>>,
    pub style: Option<String>,
}

impl Validatable for ContainerElement {
    fn validate(&self) -> Result<()> {
        for child in &self.children {
            child.validate()?;
        }
        Ok(())
    }
}

impl Renderable for ContainerElement {
    fn render(&self, indent: usize) -> Result<()> {
        let indent_str = " ".repeat(indent);
        let style = self.style.as_ref().map_or("".to_string(), |s| format!(" [стиль: {}]", s));
        println!("{}{}{}", indent_str, "container", style);
        for child in &self.children {
            child.render(indent + 2)?;
        }
        Ok(())
    }
}

// --- Enum для десериализации JSON ---
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum DeserUIElement {
    #[serde(rename = "text")]
    Text {
        content: String,
        #[serde(default)]
        style: Option<String>,
        #[serde(default)]
        on_click: Option<String>,
    },
    #[serde(rename = "button")]
    Button {
        label: String,
        action: String,
        #[serde(default)]
        style: Option<String>,
        #[serde(default)]
        on_click: Option<String>,
    },
    #[serde(rename = "container")]
    Container {
        #[serde(default)]
        children: Vec<DeserUIElement>,
        #[serde(default)]
        style: Option<String>,
    },
}

impl DeserUIElement {
    pub fn to_component(self) -> Box<DynComponent> {
        match self {
            DeserUIElement::Text { content, style, on_click } => {
                Box::new(TextElement { content, style, on_click })
            }
            DeserUIElement::Button { label, action, style, on_click } => {
                Box::new(ButtonElement { label, action, style, on_click })
            }
            DeserUIElement::Container { children, style } => {
                let dyn_children: Vec<Box<DynComponent>> = 
                    children.into_iter().map(|c| c.to_component()).collect();
                Box::new(ContainerElement { children: dyn_children, style })
            }
        }
    }
}

// --- Остальные структуры ---
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct UserData {
    pub age: u32,
    pub preferences: Vec<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct UIDescription {
    pub elements: Vec<DeserUIElement>,
    pub user_data: UserData,
}

// --- Генерация UI ---
pub fn generate_ui(elements: &[Box<DynComponent>], order: &[usize], indent: usize) -> Result<()> {
    for &index in order {
        let element = &elements[index];
        element.render(indent)?;
    }
    Ok(())
}