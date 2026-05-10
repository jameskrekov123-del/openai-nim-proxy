// server.js - OpenAI to NVIDIA NIM API Proxy (Stable 2026 Build)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = true;
const ENABLE_THINKING_MODE = true; 

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'z-ai/glm-5.1', // Switched to GLM 5.1 as requested
  'glm-5.1': 'z-ai/glm-5.1',
  'v4-pro': 'deepseek-ai/deepseek-v4-pro'
};

app.post('/v1/chat/completions', async (req, res) => {
  try {
    let { model, messages, temperature, max_tokens, stream } = req.body;
    let nimModel = MODEL_MAPPING[model] || model;
    const isGLM = nimModel.includes('glm');

    // --- GLM 5.1 PARAGRAPH & THINKING FIX ---
    let extraKwargs = {};
    if (ENABLE_THINKING_MODE) {
      if (isGLM) {
        // 2026 GLM 5.1 Double-Key Logic to prevent API hangs
        extraKwargs = {
          chat_template_kwargs: {
            enable_thinking: true, // Specific for GLM
            thinking: true,        // Redundant but required by NIM v1.4+
            clear_thinking: false, // Set to false so you can see it
            do_sample: true
          }
        };
        // Inject formatting guard to stop the "Wall of Text"
        messages.push({
            role: "system", 
            content: "[INSTRUCTION: Separate paragraphs with double newlines. Do not output a single block of text.]"
        });
      } else if (nimModel.includes('deepseek') || nimModel.includes('thinking')) {
        extraKwargs = {
          chat_template_kwargs: { thinking: true, reasoning_effort: "high" }
        };
      }
    }

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: isGLM ? 1.0 : (temperature || 0.8), 
      max_tokens: max_tokens || 8192,
      stream: stream || false,
      ...extraKwargs
    };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json',
      timeout: 30000 // 30 second timeout to prevent "Suspicious Builds" from hanging
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      let isThinking = false;
      let buffer = ''; // CRITICAL: Restored the buffer to handle split chunks

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep the incomplete line for the next chunk

        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                let content = data.choices[0].delta.content || '';
                const reasoning = data.choices[0].delta.reasoning_content || '';

                if (SHOW_REASONING) {
                  if (reasoning) {
                    if (!isThinking) {
                      isThinking = true;
                      content = '<think>\n' + reasoning;
                    } else { content = reasoning; }
                  } else if (isThinking && content) {
                    isThinking = false;
                    content = '\n</think>\n\n' + content;
                  }
                }
                data.choices[0].delta.content = content;
                delete data.choices[0].delta.reasoning_content;
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) { /* silent catch for partial lines */ }
          }
        });
      });
      response.data.on('end', () => res.end());
    } else {
      res.json(response.data);
    }
  } catch (error) {
    console.error('Proxy error:', error.response?.data || error.message);
    res.status(500).json({ error: { message: "NIM API error or timeout. Check your API Key." } });
  }
});

// Health check to satisfy build-platform requirements
app.get('/health', (req, res) => res.status(200).send('OK'));

app.listen(PORT, '0.0.0.0', () => console.log(`Server running on port ${PORT}`));
