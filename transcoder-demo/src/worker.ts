import init, { TranscoderWasm, type Message } from "transcoder-wasm";
import wasmUrl from "transcoder-wasm/transcoder_wasm_bg.wasm?url";

export type WorkerMessage = {
	data: Uint8Array;
};

async function initWorker() {
	await init(wasmUrl);

	const transcoder = new TranscoderWasm();

	let busy = false;

	self.onmessage = async (ev) => {
		if (busy) {
			console.error("worker: busy?");
			return;
		}

		busy = true;

		let msg = ev.data as WorkerMessage;

		console.log("worker: message:", msg);

		transcoder.transcode(msg.data, (m: Message) => self.postMessage(m));

		busy = false;
	};

	console.log("worker: inited");
}

initWorker();
