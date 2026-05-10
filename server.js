// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Bumped to 50MB for safer handling of long contexts
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = true;

// 🔥 THINKING MODE TOGGLE
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
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash',
  'glm-5.1': 'z-ai/glm-5.1',
  'v4-pro': 'deepseek-ai/deepseek-v4-pro',
  'v4-flash': 'deepseek-ai/deepseek-v4-flash'
};

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Chat completions
app.post('/v1/chat/completions', async (req, res) => {
  try {
    let { model, messages, temperature, max_tokens, stream } = req.body;
    let nimModel = MODEL_MAPPING[model] || model;

    // --- REASONING RECONSTRUCTION ---
    // This looks at previous messages. If it finds a <think> block, 
    // it moves it to reasoning_content so the model doesn't get confused.
    const processedMessages = messages.map(msg => {
      if (msg.role === 'assistant' && msg.content && msg.content.includes('<think>')) {
        const parts = msg.content.split('</think>');
        return {
          role: 'assistant',
          content: parts[1]?.trim() || '',
          reasoning_content: parts[0].replace('<think>', '').trim()
        };
      }
      return msg;
    });

    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature || 0.8, // Note: Ignored by V4 when thinking
      max_tokens: max_tokens || 8192,  // Bumped default for long RP posts
      stream: stream || false,
      ...(ENABLE_THINKING_MODE && {
        chat_template_kwargs: { 
          thinking: true,
          // "high" is better for RP than "max". Max is for math/coding.
          reasoning_effort: "high" 
        }
      })
    };

    // ... rest of your axios call and streaming logic ...
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let isThinking = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

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

                // Clean tokens
                content = content.replace(/<\|start_header_id\|>assistant<\|end_header_id\|>/g, '')
                                 .replace(/<\|start_header_id\|>.*?<\|end_header_id\|>/g, '');

                if (SHOW_REASONING) {
                  if (reasoning) {
                    if (!isThinking) {
                      isThinking = true;
                      content = '<think>\n' + reasoning;
                    } else {
                      content = reasoning;
                    }
                  } else if (isThinking && content) {
                    isThinking = false;
                    content = '\n</think>\n\n' + content;
                  }
                }

                data.choices[0].delta.content = content;
                if (data.choices[0].delta.reasoning_content) {
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });

      response.data.on('end', () => res.end());
    } else {
      // Non-streaming logic
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          return {
            index: choice.index,
            message: { role: choice.message.role, content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage
      };
      res.json(openaiResponse);
    }
  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(500).json({ error: { message: error.message } });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on port ${PORT}`);
});
