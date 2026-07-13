const API_BASE = '/api'

export function useApi() {
  async function fetchModels(backend = null) {
    const qs = backend ? `?backend=${encodeURIComponent(backend)}` : ''
    const resp = await fetch(`${API_BASE}/models${qs}`)
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

  function streamResearch(query, model, sessionId, callbacks, prioritizeKb = false, backend = null) {
    const body = JSON.stringify({
      query,
      model: model || null,
      backend: backend || null,
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

  async function fetchConversationTranscript(id) {
    const resp = await fetch(`${API_BASE}/kb/topics/${id}/conversation`)
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

  async function triggerTopicVerification(id) {
    const resp = await fetch(`${API_BASE}/kb/topics/${id}/verify`, { method: 'POST' })
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}))
      throw new Error(body.detail || 'Failed to start verification')
    }
    return resp.json()
  }

  async function fetchTopicProcessingStatus(id) {
    const resp = await fetch(`${API_BASE}/kb/topics/${id}/processing`)
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

  async function fetchResolutionCandidates(type = null, status = 'open') {
    const params = new URLSearchParams({ status })
    if (type) params.set('type', type)
    const resp = await fetch(`${API_BASE}/kb/resolution-candidates?${params}`)
    return resp.json()
  }

  async function reviewResolutionCandidate(candidateId, decision) {
    const resp = await fetch(`${API_BASE}/kb/resolution-candidates/${candidateId}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision }),
    })
    return resp.json()
  }

  // --- Knowledge base (sources: ingest/chunk/extract/verify) ---

  async function fetchSources(q = '', limit = 50) {
    const params = new URLSearchParams({ q, limit })
    const resp = await fetch(`${API_BASE}/kb/sources?${params}`)
    return resp.json()
  }

  async function fetchSource(id) {
    const resp = await fetch(`${API_BASE}/kb/sources/${id}`)
    if (!resp.ok) throw new Error('Source not found')
    return resp.json()
  }

  async function fetchSourceClaims(id) {
    const resp = await fetch(`${API_BASE}/kb/sources/${id}/claims`)
    return resp.json()
  }

  async function fetchSourceProcessingStatus(id) {
    const resp = await fetch(`${API_BASE}/kb/sources/${id}/processing`)
    return resp.json()
  }

  async function ingestUrl(url, trustTier = null) {
    const resp = await fetch(`${API_BASE}/kb/sources/ingest-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, trust_tier: trustTier }),
    })
    return resp.json()
  }

  async function ingestYoutube(url, trustTier = null) {
    const resp = await fetch(`${API_BASE}/kb/sources/ingest-youtube`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, trust_tier: trustTier }),
    })
    return resp.json()
  }

  async function ingestConversation(text, title = null, trustTier = null, topicName = null) {
    const resp = await fetch(`${API_BASE}/kb/sources/ingest-conversation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, title, trust_tier: trustTier, topic_name: topicName }),
    })
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}))
      throw new Error(body.detail || 'Failed to ingest conversation')
    }
    return resp.json()
  }

  async function ingestFile(file, trustTier = null) {
    const formData = new FormData()
    formData.append('file', file)
    if (trustTier) formData.append('trust_tier', trustTier)
    const resp = await fetch(`${API_BASE}/kb/sources/ingest-file`, {
      method: 'POST',
      body: formData,
    })
    return resp.json()
  }

  async function chunkSource(id, chunkSize = 1200) {
    const resp = await fetch(`${API_BASE}/kb/sources/${id}/chunk`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chunk_size: chunkSize }),
    })
    return resp.json()
  }

  async function extractSource(id, force = false) {
    const resp = await fetch(`${API_BASE}/kb/sources/${id}/extract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force }),
    })
    return resp.json()
  }

  async function verifySource(id, force = false, threshold = null) {
    const resp = await fetch(`${API_BASE}/kb/sources/${id}/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force, threshold }),
    })
    return resp.json()
  }

  async function backfillEmbeddings() {
    const resp = await fetch(`${API_BASE}/kb/embeddings/backfill`, { method: 'POST' })
    return resp.json()
  }

  // --- Knowledge base (claims: browse/verify/search) ---

  async function fetchClaims(limit = 100) {
    const resp = await fetch(`${API_BASE}/kb/claims?limit=${limit}`)
    return resp.json()
  }

  async function fetchClaim(id) {
    const resp = await fetch(`${API_BASE}/kb/claims/${id}`)
    if (!resp.ok) throw new Error('Claim not found')
    return resp.json()
  }

  async function verifyClaim(id, force = false) {
    const resp = await fetch(`${API_BASE}/kb/claims/${id}/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force }),
    })
    return resp.json()
  }

  async function setPreferredSource(claimId, sourceId) {
    const resp = await fetch(`${API_BASE}/kb/claims/${claimId}/preferred-source`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_id: sourceId }),
    })
    return resp.json()
  }

  async function setClaimVerificationOverride(claimId, override) {
    const resp = await fetch(`${API_BASE}/kb/claims/${claimId}/verification-override`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ override }),
    })
    return resp.json()
  }

  async function setClaimVerificationContext(claimId, context) {
    const resp = await fetch(`${API_BASE}/kb/claims/${claimId}/verification-context`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ context }),
    })
    return resp.json()
  }

  async function searchChunks(q, semantic = false, limit = 20) {
    const params = new URLSearchParams({ q, semantic, limit })
    const resp = await fetch(`${API_BASE}/kb/search?${params}`)
    return resp.json()
  }

  // --- Knowledge base (verification runs: history/current/trigger) ---

  async function fetchVerificationRuns(limit = 30) {
    const resp = await fetch(`${API_BASE}/kb/verification-runs?limit=${limit}`)
    return resp.json()
  }

  async function fetchCurrentVerificationRun() {
    const resp = await fetch(`${API_BASE}/kb/verification-runs/current`)
    return resp.json()
  }

  async function triggerVerificationRun(threshold = null, force = false) {
    const resp = await fetch(`${API_BASE}/kb/verification-runs/trigger`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ threshold, force }),
    })
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}))
      throw new Error(body.detail || 'Failed to start verification run')
    }
    return resp.json()
  }

  // --- Search provider usage ---

  async function fetchSearchUsage() {
    const resp = await fetch(`${API_BASE}/search-usage`)
    return resp.json()
  }

  async function checkSearchProviders() {
    const resp = await fetch(`${API_BASE}/search-usage/check`, { method: 'POST' })
    return resp.json()
  }

  return {
    fetchModels, fetchSessions, fetchSession, deleteSession, streamResearch,
    fetchTopics, createTopic, fetchTopic, fetchTimeline, fetchTopicClaims,
    fetchConversationTranscript,
    fetchTopicSources, reviewClaimSuggestion, reviewSourceSuggestion,
    backfillTopic, triggerTopicVerification, fetchTopicProcessingStatus, fetchReport, generateReport,
    fetchResolutionCandidates, reviewResolutionCandidate,
    fetchSources, fetchSource, fetchSourceClaims, fetchSourceProcessingStatus,
    ingestUrl, ingestYoutube, ingestFile, ingestConversation,
    chunkSource, extractSource, verifySource, backfillEmbeddings,
    fetchClaims, fetchClaim, verifyClaim, setPreferredSource, setClaimVerificationOverride,
    setClaimVerificationContext, searchChunks,
    fetchVerificationRuns, fetchCurrentVerificationRun, triggerVerificationRun,
    fetchSearchUsage, checkSearchProviders,
  }
}
