import os
import io
import time
import struct
import uvicorn
import argparse
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import soundfile as sf

from loguru import logger
logger.add('logs/api_server_v2.log', rotation='10 MB', retention=10, level='DEBUG', enqueue=True)

from indextts.infer_vllm_v2 import IndexTTS2

tts = None
def wav_header(sr: int, ch: int, bits_per_sample: int = 16, data_size: int = 0xFFFFFFFF) -> bytes:
    byte_rate = sr * ch * (bits_per_sample // 8)
    block_align = ch * (bits_per_sample // 8)

    # RIFF chunk size is 36 + data_size, but we cannot know data_size in streaming.
    riff_size = 36 + data_size
    if riff_size > 0xFFFFFFFF:
        riff_size = 0xFFFFFFFF

    return b''.join([
        b'RIFF',
        struct.pack('<I', riff_size & 0xFFFFFFFF),
        b'WAVE',
        b'fmt ',
        struct.pack('<I', 16),                     # PCM fmt chunk size
        struct.pack('<H', 1),                      # PCM format
        struct.pack('<H', ch),
        struct.pack('<I', sr),
        struct.pack('<I', byte_rate),
        struct.pack('<H', block_align),
        struct.pack('<H', bits_per_sample),
        b'data',
        struct.pack('<I', data_size & 0xFFFFFFFF), # unknown, set large
    ])

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
    )
    yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allows all origins, change in production for security
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/health')
async def health_check():
    '''Health check endpoint'''
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                'status': 'unhealthy',
                'message': 'TTS model not initialized'
            }
        )
    
    return JSONResponse(
        status_code=200,
        content={
            'status': 'healthy',
            'message': 'Service is running',
            'timestamp': time.time()
        }
    )

@app.post("/tts_url")
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value

        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
            vec = None
        elif emo_control_method == 1:
            vec = None
        elif emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "error": "情感向量之和不能超过1.5，请调整后重试。"},
                )
        else:
            vec = None

        async def gen():
            # Stream WAV header first, then PCM16 chunks
            sr = 22050
            ch = 1
            yield wav_header(sr, ch, 16)

            async for sr_out, pcm16 in tts.infer_stream(
                spk_audio_prompt=spk_audio_path,
                text=text,
                emo_audio_prompt=emo_ref_path,
                emo_alpha=emo_weight,
                emo_vector=vec,
                use_emo_text=(emo_control_method == 3),
                emo_text=emo_text,
                use_random=emo_random,
                max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
            ):
                yield pcm16

        return StreamingResponse(
            gen(),
            media_type="audio/wav",
            headers={
                "X-Audio-Sample-Rate": "22050",
                "X-Audio-Channels": "1",
                "X-Audio-Format": "s16le",
            },
        )

    except Exception as ex:
        tb_str = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(status_code=500, content={"status": "error", "error": str(tb_str)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--model_dir', type=str, default='checkpoints/IndexTTS-2-vLLM', help='Model checkpoints directory')
    parser.add_argument('--is_fp16', action='store_true', default=False, help='Fp16 infer')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.25)
    parser.add_argument('--qwenemo_gpu_memory_utilization', type=float, default=0.10)
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose mode')
    args = parser.parse_args()
    
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    uvicorn.run(app=app, host=args.host, port=args.port)