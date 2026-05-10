// server.js - OpenAI to NVIDIA NIM API Proxy
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
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro', 
  'glm-5.1': 'z-ai/glm-5.1',
  'v4-pro': 'deepseek-ai/deepseek-v4-pro'
};

app.post('/v1/chat/completions', async (req, res) => {
  try {
    let { model, messages, temperature, max_tokens, stream } = req.body;
    let nimModel = MODEL_MAPPING[model] || model;
    const isGLM = nimModel.includes('glm');

    // --- GLM PARAGRAPH FIX ---
    // Injects a reminder at the end of the prompt to stop the "one paragraph" collapse.
    if (isGLM) {
        messages.push({
            role: "system", 
            content: "[FORMATTING: Ensure you use double newlines (\\n\\n) between every paragraph. Do not collapse your response into a single block of text.]"
        });
    }

    // --- MODEL SPECIFIC THINKING LOGIC ---
    let extraKwargs = {};
    if (ENABLE_THINKING_MODE) {
      if (isGLM) {
        // GLM 5.1 requires these specific keys to show the "Thinking Box"
        extraKwargs = {
          chat_template_kwargs: {
            enable_thinking: true,
            clear_thinking: false,
            do_sample: true
          }
        };
      } else if (nimModel.includes('deepseek') || nimModel.includes('thinking')) {
        extraKwargs = {
          chat_template_kwargs: { thinking: true, reasoning_effort: "high" }
        };
      }
    }

    const nimRequest = {
      model: nimModel,
      messages: messages,
      // GLM 5.1 prefers Temperature 1.0 to maintain paragraph structure
      temperature: isGLM ? 1.0 : (temperature || 0.8), 
      max_tokens: max_tokens || 8192,
      stream: stream || false,
      ...extraKwargs
    };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      let isThinking = false;

      response.data.on('data', (chunk) => {
        const lines = chunk.toString().split('\n');
        lines.forEach(line => {
          if (line.startsWith('data: ') && !line.includes('[DONE]')) {
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
            } catch (e) { /* skip malformed */ }
          } else if (line.includes('[DONE]')) {
            res.write(line + '\n');
          }
        });
      });
      response.data.on('end', () => res.end());
    } else {
      // Non-streaming logic (standard JSON response)
      res.json(response.data);
    }
  } catch (error) {
    console.error('Proxy error:', error.response?.data || error.message);
    res.status(500).json({ error: { message: error.message } });
  }
});

app.listen(PORT, '0.0.0.0', () => console.log(`Server running on port ${PORT}`));
