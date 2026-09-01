/**
 * Cloudflare Worker for Voice Cloning DJ
 * Proxies RunPod API requests to bypass CORS issues
 * 
 * Deploy to Cloudflare Workers:
 * 1. Go to workers.cloudflare.com
 * 2. Create new worker
 * 3. Paste this code
 * 4. Deploy
 * 5. Use worker URL in frontend
 */

const RUNPOD_ENDPOINT = 'tat2r9q5eh58vy';
const RUNPOD_API = 'https://api.runpod.io/v2';

export default {
  async fetch(request) {
    // Handle CORS
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      });
    }

    const url = new URL(request.url);
    const path = url.pathname;

    // Route: POST /run - Start job
    if (path === '/run' && request.method === 'POST') {
      return handleRun(request);
    }

    // Route: GET /status/:jobId - Get job status
    if (path.startsWith('/status/') && request.method === 'GET') {
      const jobId = path.split('/')[2];
      return handleStatus(jobId);
    }

    // Route: GET /health - Health check
    if (path === '/health') {
      return new Response(JSON.stringify({ status: 'ok' }), {
        headers: { 'Content-Type': 'application/json' },
      });
    }

    return new Response('Not Found', { status: 404 });
  },
};

async function handleRun(request) {
  try {
    const payload = await request.json();

    // Validate payload
    if (!payload.mode || !payload.text_to_synthesize) {
      return errorResponse('Missing required fields', 400);
    }

    // Forward to RunPod
    const response = await fetch(`${RUNPOD_API}/${RUNPOD_ENDPOINT}/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    return corsResponse(data);
  } catch (error) {
    return errorResponse(error.message);
  }
}

async function handleStatus(jobId) {
  try {
    const response = await fetch(`${RUNPOD_API}/${RUNPOD_ENDPOINT}/status/${jobId}`);
    const data = await response.json();

    return corsResponse(data);
  } catch (error) {
    return errorResponse(error.message);
  }
}

function corsResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}

function errorResponse(message, status = 500) {
  return corsResponse({ error: message }, status);
}
