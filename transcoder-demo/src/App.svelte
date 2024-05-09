<script lang="ts">
	import type { Message, Metadata } from "transcoder-wasm";
	import type { WorkerMessage } from "./worker";
	import TranscoderWorker from "./worker.ts?worker";

	let working = false;
	let state = "";
	let progress = 0;
	let metadata: Metadata | null = null;

	const downloadFile = (filename: string, bytes: Uint8Array) => {
		let blob = new Blob([bytes], { type: "audio/ogg" });
		let link = document.createElement("a");
		link.href = window.URL.createObjectURL(blob);
		link.download = filename;
		link.click();
	};

	const worker = new TranscoderWorker();
	worker.addEventListener("error", () => {
		console.error("main: worker errored");
		working = false;
	});
	worker.addEventListener("message", (ev) => {
		let msg = ev.data as Message;

		if (msg.type == "loading") {
			working = true;
			state = "loading";
			progress = 0;
			metadata = null;
		} else if (msg.type == "resampling") {
			working = true;
			state = "resampling";
			progress = msg.data;
			metadata = null;
		} else if (msg.type == "encoding") {
			working = true;
			state = "encoding";
			progress = msg.data;
			metadata = null;
		} else if (msg.type == "finish") {
			working = false;
			state = "";
			progress = 0;
			metadata = msg.data.metadata;
			downloadFile("test.ogg", msg.data.data);
		} else if (msg.type == "error") {
			working = false;
			state = "";
			progress = 0;
			metadata = null;
		}
	});

	let inputRef: HTMLInputElement;

	const onChange = async () => {
		let files = inputRef.files ?? [];

		let file = files[0];

		if (!file) {
			return;
		}

		let buf = await file.arrayBuffer();
		let uint8arr = new Uint8Array(buf);

		worker.postMessage({ data: uint8arr } satisfies WorkerMessage);
	};
</script>

<main>
	<h1>Wasm Opus Transcoder Demo</h1>

	<input
		type="file"
		disabled={working}
		bind:this={inputRef}
		on:change={onChange}
	/>

	{#if working}
		<div class="progress">
			{state}
			{#if progress > 0}
				({Math.floor(progress * 100)}%)
			{/if}
		</div>
	{/if}

	{#if metadata}
		<div class="metadata">
			{#if metadata.art}
				{@const blob = new Blob([metadata.art.data_2048], {
					type: "image/jpeg",
				})}
				{@const imageUrl = URL.createObjectURL(blob)}
				<div class="image">
					<img
						width={128}
						height={128}
						src={imageUrl}
						alt="Cover art"
					/>
				</div>
			{:else}
				<div class="image empty">(no art)</div>
			{/if}

			<div class="details">
				<div>
					{metadata.title ?? ""}
				</div>
				<div>
					{metadata.artist ?? ""}
				</div>
				<div>
					{metadata.album ?? ""}
				</div>
				<div>
					{Math.floor(metadata.duration / 60)}:{(
						Math.floor(metadata.duration) % 60
					)
						.toString()
						.padStart(2, "0")}
				</div>
			</div>
		</div>
	{/if}

	<a href="https://github.com/hazelmeow/wasm-opus-transcoder" target="_blank">
		source code
	</a>
</main>

<style>
	main {
		display: flex;
		flex-direction: column;
		place-items: center;
		gap: 16px;
	}

	h1 {
		font-size: 24px;
		margin: 0;
	}

	.progress {
		height: 128px;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.image {
		width: 128px;
		height: 128px;
	}

	.empty {
		border: 2px solid #ccc;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.metadata {
		display: flex;
		flex-direction: row;
		gap: 16px;
	}

	.details {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
</style>
