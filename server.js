// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const http = require('http');
const https = require('https');
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = true;
const ENABLE_THINKING_MODE = true;

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'glm-4.7': 'z-ai/glm4_7',
  'z-ai/glm4_7': 'z-ai/glm4_7',
  'glm-5.1': 'z-ai/glm-5.1',
  'z-ai/glm-5.1': 'z-ai/glm-5.1',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro',
  'v4-pro': 'deepseek-ai/deepseek-v4-pro'
};

app.post('/v1/chat/completions', async (req, res) => {
  try {
    let { model, messages, temperature, max_tokens, stream } = req.body;
    let nimModel = MODEL_MAPPING[model] || model;

    // Injection to force paragraphs for GLM-5.1
    const processedMessages = messages.map((msg, index) => {
        if (index === messages.length - 1 && msg.role === 'user') {
            return { ...msg, content: msg.content + "\n\n(Format your response with clear double-spaced paragraphs. Do not use a wall of text.)" };
        }
        return msg;
    });

    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature || 0.8,
      max_tokens: max_tokens || 8192,
      stream: stream || false,
      ...(ENABLE_THINKING_MODE && {
        chat_template_kwargs: { enable_thinking: true, clear_thinking: false },
        reasoning_effort: "medium" 
      })
    };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json',
      timeout: 300000, 
      httpAgent: new http.Agent({ keepAlive: true }),
      httpsAgent: new https.Agent({ keepAlive: true })
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      let buffer = '';
      let isThinking = false;
      let hasStartedDialogue = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) { res.write(line + '\n'); return; }

            try {
              const data = JSON.parse(line.slice(6));
              const delta = data.choices[0].delta;
              let content = delta.content || '';
              let reasoning = delta.reasoning_content || '';
              let finalOutput = '';

              if (SHOW_REASONING) {
                // 1. Handle incoming reasoning stream
                if (reasoning) {
                  if (!isThinking) {
                    isThinking = true;
                    finalOutput += '<think>\n';
                  }
                  finalOutput += reasoning;
                }
                
                // 2. Transition from reasoning to dialogue
                if (isThinking && content && !reasoning) {
                  isThinking = false;
                  hasStartedDialogue = true;
                  finalOutput += '\n</think>\n\n';
                }
              }

              // 3. HARD FILTER: Prevent the model from typing its own <think> tags in the dialogue
              if (content) {
                let cleanContent = content
                  .replace(/<think>|<\/think>/gi, '') // Wipe leaked tags
                  .replace(/<\|start_header_id\|>.*?<\|end_header_id\|>/g, '');
                
                finalOutput += cleanContent;
              }

              if (finalOutput) {
                data.choices[0].delta.content = finalOutput;
                delete data.choices[0].delta.reasoning_content;
                res.write(`data: ${JSON.stringify(data)}\n\n`);
              }
            } catch (e) {}
          }
        });
      });
      response.data.on('end', () => res.end());
    } else {
        // Non-streaming logic (standardized)
        let choice = response.data.choices[0];
        let fullText = choice.message.content.replace(/<think>|<\/think>/gi, '');
        if (SHOW_REASONING && choice.message.reasoning_content) {
            fullText = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${fullText}`;
        }
        res.json({
            ...response.data,
            choices: [{ ...choice, message: { ...choice.message, content: fullText } }]
        });
    }
  } catch (error) {
    res.status(500).json({ error: { message: error.message } });
  }
});

app.listen(PORT, '0.0.0.0', () => console.log(`Proxy active on ${PORT}`));
