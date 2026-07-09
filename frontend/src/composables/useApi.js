const API_BASE = '/api'

export function useApi() {
  async function fetchModels() {
    const resp = await fetch(`${API_BASE}/models`)
    return resp.json()
  }

  async function fetchSessions(limit = 50) {
    const resp = await fetch(`${API_BASE}/sessions?limit=${limit}`)
    return resp.json()
  }

  async function fetchSession(id) {
    const resp = await fetch(`${API_BASE}/sessions/${id}`)
    if (!resp.ok) throw new Error('Session not found')
    return resp.json()
  }

  async function deleteSession(id) {
    const resp = await fetch(`${API_BASE}/sessions/${id}`, { method: 'DELETE' })
    return resp.json()
  }

  function streamResearch(query, model, sessionId, callbacks, prioritizeKb = false) {
    const body = JSON.stringify({
      query,
      model: model || null,
      session_id: sessionId || null,
      prioritize_kb: !!prioritizeKb,
    })

    const controller = new AbortController()

    fetch(`${API_BASE}/research`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      signal: controller.signal,
    }).then(async (resp) => {
      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let currentEvent = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          // Skip comments/pings
          if (line.startsWith(':') || line.trim() === '') {
            continue
          }

          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim()
            continue
          }

          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            try {
              const parsed = JSON.parse(data)

              switch (currentEvent) {
                case 'session':
                  callbacks.onSession?.(parsed)
                  break
                case 'status':
                  callbacks.onStatus?.(parsed)
                  break
                case 'tool':
                  callbacks.onTool?.(parsed)
                  break
                case 'answer':
                  callbacks.onAnswer?.(parsed)
                  break
                case 'error':
                  callbacks.onError?.(parsed)
                  break
                case 'done':
                  // Stream complete
                  break
              }
            } catch (e) {
              // Not JSON, skip
            }
            currentEvent = null
          }
        }
      }

      callbacks.onDone?.()
    }).catch((err) => {
      if (err.name !== 'AbortError') {
        callbacks.onError?.({ error: err.message })
      }
    })

    return controller
  }

  // --- Knowledge base (topics/timeline/reports) ---

  async function fetchTopics() {
    const resp = await fetch(`${API_BASE}/kb/topics`)
    return resp.json()
  }

  async function createTopic(name, description) {
    const resp = await fetch(`${API_BASE}/kb/topics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description: description || null }),
    })
    return resp.json()
  }

  async function fetchTopic(id) {
    const resp = await fetch(`${API_BASE}/kb/topics/${id}`)
    if (!resp.ok) throw new Error('Topic not found')
    return resp.json()
  }

  async function fetchTimeline(id) {
    const resp = await fetch(`${API_BASE}/kb/topics/${id}/timeline`)
    return resp.json()
  }

  async function fetchTopicClaims(id, status = 'attached') {
    const resp = await fetch(`${API_BASE}/kb/topics/${id}/claims?status=${status}`)
    return resp.json()
  }

  async function fetchTopicSources(id, status = 'attached') {
    const resp = await fetch(`${API_BASE}/kb/topics/${id}/sources?status=${status}`)
    return resp.json()
  }

  async function reviewClaimSuggestion(topicId, claimId, decision) {
    const resp = await fetch(`${API_BASE}/kb/topics/${topicId}/claims/${claimId}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision }),
    })
    return resp.json()
  }

  async function reviewSourceSuggestion(topicId, sourceId, decision) {
    const resp = await fetch(`${API_BASE}/kb/topics/${topicId}/sources/${sourceId}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision }),
    })
    return resp.json()
  }

  async function backfillTopic(id) {
    const resp = await fetch(`${API_BASE}/kb/topics/${id}/backfill`, { method: 'POST' })
    return resp.json()
  }

  async function fetchReport(id) {
    const resp = await fetch(`${API_BASE}/kb/topics/${id}/report`)
    return resp.json()
  }

  async function generateReport(id) {
    const resp = await fetch(`${API_BASE}/kb/topics/${id}/report`, { method: 'POST' })
    return resp.json()
  }

  return {
    fetchModels, fetchSessions, fetchSession, deleteSession, streamResearch,
    fetchTopics, createTopic, fetchTopic, fetchTimeline, fetchTopicClaims,
    fetchTopicSources, reviewClaimSuggestion, reviewSourceSuggestion,
    backfillTopic, fetchReport, generateReport,
  }
}
