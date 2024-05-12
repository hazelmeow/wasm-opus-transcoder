# Wasm Opus Transcoder

Encoding to Opus uses [unsafe-libopus](https://github.com/DCNick3/unsafe-libopus)
and [this patch](https://github.com/DCNick3/opus-rs/tree/unsafe-libopus)
(actually [my fork](https://github.com/hazelmeow/opus-rs/tree/unsafe-libopus))
for [opus-rs](https://github.com/SpaceManiac/opus-rs).
Then we glue that together with [Symphonia](https://github.com/pdeljanov/Symphonia) for pure Rust audio decoding,
[Rubato](https://github.com/HEnquist/rubato) for resampling,
and [ogg](https://github.com/RustAudio/ogg).
You can see this in `transcoder/`.

`transcoder-wasm/` is a simple wrapper that provides JS bindings which gets compiled with `wasm-pack`.

There's a Svelte demo in `transcoder-demo/`.
The transcoding is done in a web worker that comunicates with the UI to prevent it from blocking the page while it's working.
You can try it online [here](https://hazelmeow.github.io/wasm-opus-transcoder/).
