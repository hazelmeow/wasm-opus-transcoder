use transcoder::{Art, TranscodeOptions, TranscodeOutput};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    fn alert(s: &str);
}

macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen(start)]
pub fn init() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    console_log!("transcoder-wasm initialized");
}

#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND_CONTENT: &'static str = r#"
export type Message =
  | { type: "loading", data: null }
  | { type: "resampling", data: number }
  | { type: "encoding", data: number }
  | { type: "finish", data: TranscodeOutput }
  | { type: "error", data: string };
"#;

enum Message {
    Loading,
    Resampling(f32),
    Encoding(f32),
    Finish(TranscodeOutput),
    Error(String),
}

impl Message {
    fn into_js(self) -> js_sys::Object {
        let obj = js_sys::Object::new();

        let ty = match self {
            Message::Loading => "loading",
            Message::Resampling(_) => "resampling",
            Message::Encoding(_) => "encoding",
            Message::Finish(_) => "finish",
            Message::Error(_) => "error",
        };

        let _ = js_sys::Reflect::set(&obj, &"type".into(), &ty.into());

        let data = match self {
            Message::Loading => JsValue::null(),
            Message::Resampling(val) => val.into(),
            Message::Encoding(val) => val.into(),
            Message::Finish(data) => {
                serde_wasm_bindgen::to_value(&data).unwrap_or_else(|_| JsValue::null())
            }
            Message::Error(error) => error.into(),
        };

        let _ = js_sys::Reflect::set(&obj, &"data".into(), &data);

        obj
    }
}

impl From<transcoder::Progress> for Message {
    fn from(value: transcoder::Progress) -> Self {
        match value {
            transcoder::Progress::Loading => Self::Loading,
            transcoder::Progress::Resampling(v) => Self::Resampling(v as f32 / 100.0),
            transcoder::Progress::Encoding(v) => Self::Encoding(v as f32 / 100.0),
        }
    }
}

#[wasm_bindgen]
pub struct TranscoderWasm {}

#[wasm_bindgen]
impl TranscoderWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    pub fn hash_bytes(&self, bytes: &[u8]) -> String {
        transcoder::hash_bytes(bytes)
    }

    pub fn transcode(&self, bytes: Vec<u8>, message_callback: &js_sys::Function) {
        console_log!("transcoding {:?} bytes", bytes.len());

        let this = JsValue::null();

        let res = transcoder::transcode(bytes, TranscodeOptions::default(), |progress| {
            let _ = message_callback.call1(&this, &Message::from(progress).into_js());
        });

        match res {
            Ok(data) => {
                let _ = message_callback.call1(&this, &Message::Finish(data).into_js());
            }
            Err(e) => {
                let _ = message_callback.call1(&this, &Message::Error(e.to_string()).into_js());
            }
        };

        console_log!("finished transcoding");
    }

    pub fn process_art(&self, media_type: &str, bytes: Vec<u8>) -> Result<Art, String> {
        transcoder::process_art(media_type, bytes).map_err(|e| e.to_string())
    }
}

impl Default for TranscoderWasm {
    fn default() -> Self {
        Self::new()
    }
}
