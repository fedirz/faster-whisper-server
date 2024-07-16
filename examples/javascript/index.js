/**
 * Example provided by https://github.com/Gan-Xing in https://github.com/fedirz/faster-whisper-server/issues/26
 */
import fs from 'fs';
import WebSocket from 'ws';
import fetch from 'node-fetch';
import FormData from 'form-data';
import path from 'path';
import ffmpeg from 'fluent-ffmpeg';
import dotenv from 'dotenv';

dotenv.config();

const ffmpegPath = process.env.FFMPEG_PATH || '/usr/bin/ffmpeg';
ffmpeg.setFfmpegPath(ffmpegPath);

/**
 * Transcribe an audio file using the HTTP endpoint.
 * Supported file types include wav, mp3, webm, and other types supported by the OpenAI API.
 * I have tested with these three types.
 *
 * @param {string} filePath - Path to the audio file
 * @param {string} model - Model name
 * @param {string} language - Language code
 * @param {string} responseFormat - Response format
 * @param {string} temperature - Temperature setting
 */
async function transcribeFile(filePath, model, language, responseFormat, temperature) {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath));
    formData.append('model', model);
    formData.append('language', language);
    formData.append('response_format', responseFormat);
    formData.append('temperature', temperature);

    const response = await fetch(`${process.env.TRANSCRIPTION_API_BASE_URL}/v1/audio/transcriptions`, {
        method: 'POST',
        body: formData,
    });

    const transcription = await response.json();
    console.log('Transcription Response:', transcription);
}

/**
 * Translate an audio file using the HTTP endpoint.
 * Only English is supported for translation.
 * Currently, I am using GLM-4-9b-int8 to translate various voices.
 * I am not sure if the author can add an endpoint for custom API+Key translation.
 * I plan to package my frontend, fast-whisper-server, and vllm+glm-4-9b-int8 into one Docker container for unified deployment.
 *
 * @param {string} filePath - Path to the audio file
 * @param {string} model - Model name
 * @param {string} responseFormat - Response format
 * @param {string} temperature - Temperature setting
 */
async function translateFile(filePath, model, responseFormat, temperature) {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath));
    formData.append('model', model);
    formData.append('response_format', responseFormat);
    formData.append('temperature', temperature);

    const response = await fetch(`${process.env.TRANSLATION_API_BASE_URL}/v1/audio/translations`, {
        method: 'POST',
        body: formData,
    });

    const translation = await response.json();
    console.log('Translation Response:', translation);
}

/**
 * Send audio data over WebSocket for transcription.
 * Currently, the supported file type for transcription is PCM.
 * I am not sure if other types are supported.
 *
 * @param {string} filePath - Path to the audio file
 * @param {string} model - Model name
 * @param {string} language - Language code
 * @param {string} responseFormat - Response format
 * @param {string} temperature - Temperature setting
 */
async function sendAudioOverWebSocket(filePath, model, language, responseFormat, temperature) {
    const wsUrl = `ws://100.105.162.69:8000/v1/audio/transcriptions?model=${encodeURIComponent(model)}&language=${encodeURIComponent(language)}&response_format=${encodeURIComponent(responseFormat)}&temperature=${encodeURIComponent(temperature)}`;
    const ws = new WebSocket(wsUrl);

    ws.on('open', async () => {
        const audioBuffer = fs.readFileSync(filePath);
        ws.send(audioBuffer);
    });

    ws.on('message', (message) => {
        const response = JSON.parse(message);
        console.log('WebSocket Response:', response);
    });

    ws.on('close', () => {
        console.log('WebSocket connection closed');
    });

    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
    });
}

/**
 * Convert audio file to PCM format.
 *
 * @param {string} filePath - Path to the audio file
 * @returns {string} - Path to the converted PCM file
 */
async function convertToPcm(filePath) {
    const pcmFilePath = filePath.replace(path.extname(filePath), '.pcm');

    await new Promise((resolve, reject) => {
        ffmpeg(filePath)
            .audioChannels(1)
            .audioFrequency(16000)
            .audioCodec('pcm_s16le')
            .toFormat('s16le')
            .on('end', () => {
                console.log(`Audio file successfully converted to PCM: ${pcmFilePath}`);
                resolve(pcmFilePath);
            })
            .on('error', (error) => {
                console.error(`Error converting audio to PCM: ${error.message}`);
                reject(error);
            })
            .save(pcmFilePath);
    });

    return pcmFilePath;
}

async function main() {
    const model = 'Systran/faster-whisper-large-v3';
    const language = 'en';
    const responseFormat = 'json';
    const temperature = '0';
    const filePath = './path/to/your/audio.webm';  // Replace with the actual file path

    // Convert the audio file to PCM format
    const pcmFilePath = await convertToPcm(filePath);

    // Transcribe the audio file using the HTTP endpoint
    await transcribeFile(pcmFilePath, model, language, responseFormat, temperature);

    // Translate the audio file using the HTTP endpoint
    await translateFile(pcmFilePath, model, responseFormat, temperature);

    // Transcribe the audio file using the WebSocket endpoint
    await sendAudioOverWebSocket(pcmFilePath, model, language, responseFormat, temperature);
}

// Make sure to use ffmpeg version 7 or above. The default apt-get install only installs version 4.x. Also, Ubuntu 22.04 or above is required to support version 7.x.
main().catch(console.error);

// Project URL: https://github.com/Gan-Xing/whisper
