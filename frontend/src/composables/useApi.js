const API_BASE = '/api'

// Keep all API helpers honest: a JSON error body is still an HTTP failure.
// Several older callers only did `resp.json()`, which made a 400/500 look
// like malformed successful data and left users with no actionable message.
async function fetch(input, init) {
  const resp = await globalThis.fetch(input, init)
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}))
    throw new Error(body.detail || body.error || body.message || `Request failed (${resp.status})`)
  }
  return resp
}

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

  function streamResearch(query, model, sessionId, callbacks, prioritizeKb = false, researchMode = 'standard') {
    const body = JSON.stringify({
      query,
      model: model || null,
      session_id: sessionId || null,
      prioritize_kb: !!prioritizeKb,
      research_mode: researchMode,
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

  async function fetchSources(q = '', limit = 50, includeArchived = false) {
    const params = new URLSearchParams({ q, limit, include_archived: includeArchived })
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

  async function fetchSourceDecisions(id) {
    const resp = await fetch(`${API_BASE}/kb/sources/${id}/decisions`)
    return resp.json()
  }

  async function resetSourceTrustTier(id) {
    const resp = await fetch(`${API_BASE}/kb/sources/${id}/trust-tier/reset`, { method: 'POST' })
    if (!resp.ok) throw new Error('Could not reset the trust tier')
    return resp.json()
  }

  async function archiveSource(id) {
    const resp = await fetch(`${API_BASE}/kb/sources/${id}/archive`, { method: 'POST' })
    if (!resp.ok) throw new Error('Could not archive this source')
    return resp.json()
  }

  async function restoreSource(id) {
    const resp = await fetch(`${API_BASE}/kb/sources/${id}/restore`, { method: 'POST' })
    if (!resp.ok) throw new Error('Could not restore this source')
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

  async function trackPlaylist(url, trustTier = null) {
    const resp = await fetch(`${API_BASE}/kb/playlists`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, trust_tier: trustTier }),
    })
    return resp.json()
  }

  async function fetchPlaylists() {
    const resp = await fetch(`${API_BASE}/kb/playlists`)
    return resp.json()
  }

  async function fetchPlaylistVideos(id) {
    const resp = await fetch(`${API_BASE}/kb/playlists/${id}/videos`)
    return resp.json()
  }

  async function deletePlaylist(id) {
    const resp = await fetch(`${API_BASE}/kb/playlists/${id}`, { method: 'DELETE' })
    return resp.json()
  }

  async function checkPlaylist(id) {
    const resp = await fetch(`${API_BASE}/kb/playlists/${id}/check`, { method: 'POST' })
    return resp.json()
  }

  async function ingestPlaylistBatch(id, limit = null) {
    const suffix = limit ? `?limit=${encodeURIComponent(limit)}` : ''
    const resp = await fetch(`${API_BASE}/kb/playlists/${id}/ingest${suffix}`, { method: 'POST' })
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

  async function ingestTopicUrl(topicId, url, trustTier = null) {
    const resp = await fetch(`${API_BASE}/kb/topics/${topicId}/ingest-url`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, trust_tier: trustTier }),
    })
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}))
      throw new Error(body.detail || 'Failed to add source')
    }
    return resp.json()
  }

  async function ingestTopicYoutube(topicId, url, trustTier = null) {
    const resp = await fetch(`${API_BASE}/kb/topics/${topicId}/ingest-youtube`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, trust_tier: trustTier }),
    })
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}))
      throw new Error(body.detail || 'Failed to add video')
    }
    return resp.json()
  }

  async function ingestTopicFile(topicId, file, trustTier = null) {
    const formData = new FormData()
    formData.append('file', file)
    if (trustTier) formData.append('trust_tier', trustTier)
    const resp = await fetch(`${API_BASE}/kb/topics/${topicId}/ingest-file`, { method: 'POST', body: formData })
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}))
      throw new Error(body.detail || 'Failed to add file')
    }
    return resp.json()
  }

  async function cancelProcessingJob(jobId) {
    const resp = await fetch(`${API_BASE}/kb/processing-jobs/${jobId}/cancel`, { method: 'POST' })
    if (!resp.ok) throw new Error('Could not cancel this job')
    return resp.json()
  }

  async function fetchProcessingJob(jobId) {
    const resp = await fetch(`${API_BASE}/kb/processing-jobs/${jobId}`)
    return resp.json()
  }

  async function fetchProcessingJobs(limit = 50) {
    const resp = await fetch(`${API_BASE}/kb/processing-jobs?limit=${limit}`)
    return resp.json()
  }

  async function moveProcessingJob(jobId, direction) {
    const resp = await fetch(`${API_BASE}/kb/processing-jobs/${jobId}/move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ direction }),
    })
    return resp.json()
  }

  async function fetchModelExperimentProfiles() {
    const resp = await fetch(`${API_BASE}/kb/model-experiments/profiles`)
    return resp.json()
  }

  async function queueModelExperiment(payload) {
    const resp = await fetch(`${API_BASE}/kb/model-experiments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    return resp.json()
  }

  async function retryProcessingJob(jobId) {
    const resp = await fetch(`${API_BASE}/kb/processing-jobs/${jobId}/retry`, { method: 'POST' })
    if (!resp.ok) throw new Error('Could not retry this job')
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

  async function triggerAdSweep(limit = 10000) {
    const resp = await fetch(`${API_BASE}/kb/ad-sweep?limit=${limit}`, { method: 'POST' })
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

  async function fetchClaimDecisions(id) {
    const resp = await fetch(`${API_BASE}/kb/claims/${id}/decisions`)
    if (!resp.ok) throw new Error('Claim history not found')
    return resp.json()
  }

  async function findCounterEvidence(id, force = false) {
    const resp = await fetch(`${API_BASE}/kb/claims/${id}/counter-evidence?force=${force}`, { method: 'POST' })
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
    fetchSources, fetchSource, fetchSourceClaims, fetchSourceDecisions, fetchSourceProcessingStatus, resetSourceTrustTier, archiveSource, restoreSource,
    ingestUrl, ingestYoutube, ingestFile, ingestConversation, trackPlaylist, fetchPlaylists, fetchPlaylistVideos, deletePlaylist, checkPlaylist, ingestPlaylistBatch,
    ingestTopicUrl, ingestTopicYoutube, ingestTopicFile, cancelProcessingJob, fetchProcessingJob, fetchProcessingJobs, moveProcessingJob, fetchModelExperimentProfiles, queueModelExperiment, retryProcessingJob,
    chunkSource, extractSource, verifySource, backfillEmbeddings, triggerAdSweep,
    fetchClaims, fetchClaim, fetchClaimDecisions, findCounterEvidence, verifyClaim, setPreferredSource, setClaimVerificationOverride,
    setClaimVerificationContext, searchChunks,
    fetchVerificationRuns, fetchCurrentVerificationRun, triggerVerificationRun,
    fetchSearchUsage, checkSearchProviders,
  }
}
