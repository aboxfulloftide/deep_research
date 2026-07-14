<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { Globe, Plus, Search, Youtube, FileUp, Layers } from 'lucide-vue-next'
import { useApi } from '../composables/useApi.js'

const router = useRouter()
const api = useApi()

const sources = ref([])
const loading = ref(true)
const query = ref('')
const showArchived = ref(false)

const showAdd = ref(false)
const addTab = ref('url')
const urlInput = ref('')
const youtubeInput = ref('')
const playlistInput = ref('')
const fileInput = ref(null)
const trustTier = ref('')
const submitting = ref(false)
const lastIngestResult = ref(null)
const lastIngestJob = ref(null)

const showSearch = ref(false)
const searchQuery = ref('')
const searchSemantic = ref(false)
const searching = ref(false)
const searchResults = ref([])

const TRUST_TIERS = [
  { value: '', label: '(none)' },
  { value: 'official', label: 'Official' },
  { value: 'reputable_reporting', label: 'Reputable reporting' },
  { value: 'secondary_analysis', label: 'Secondary analysis' },
  { value: 'user_generated', label: 'User-generated' },
]

onMounted(async () => { await loadSources(); await loadPlaylists() })

async function loadSources() {
  loading.value = true
  const data = await api.fetchSources(query.value, 50, showArchived.value)
  sources.value = data.sources || []
  loading.value = false
}

async function loadPlaylists() {
  playlists.value = (await api.fetchPlaylists()).playlists || []
}

async function showPlaylistVideos(playlist) {
  const data = await api.fetchPlaylistVideos(playlist.id)
  playlistVideos.value = { ...playlistVideos.value, [playlist.id]: data.videos || [] }
}

async function removePlaylist(playlist) {
  await api.deletePlaylist(playlist.id)
  playlistVideos.value = Object.fromEntries(Object.entries(playlistVideos.value).filter(([id]) => id !== playlist.id))
  await loadPlaylists()
}

async function checkPlaylist(playlist) {
  const { job } = await api.checkPlaylist(playlist.id)
  for (let attempt = 0; attempt < 30; attempt++) {
    await new Promise(resolve => setTimeout(resolve, 1000))
    const status = (await api.fetchProcessingJob(job.id)).job
    if (['completed', 'partial', 'failed', 'cancelled'].includes(status.status)) break
  }
  await showPlaylistVideos(playlist)
}

async function ingestPlaylistBatch(playlist) {
  const { job } = await api.ingestPlaylistBatch(playlist.id)
  for (let attempt = 0; attempt < 30; attempt++) {
    await new Promise(resolve => setTimeout(resolve, 1000))
    const status = (await api.fetchProcessingJob(job.id)).job
    if (['completed', 'partial', 'failed', 'cancelled'].includes(status.status)) break
  }
  await showPlaylistVideos(playlist)
}

function openSource(source) {
  router.push({ name: 'source', params: { id: source.id } })
}

async function submitIngest() {
  submitting.value = true
  lastIngestResult.value = null
  lastIngestJob.value = null
  try {
    let data
    if (addTab.value === 'url') {
      if (!urlInput.value.trim()) return
      data = await api.ingestUrl(urlInput.value.trim(), trustTier.value || null)
    } else if (addTab.value === 'youtube') {
      if (!youtubeInput.value.trim()) return
      data = await api.ingestYoutube(youtubeInput.value.trim(), trustTier.value || null)
    } else if (addTab.value === 'playlist') {
      if (!playlistInput.value.trim()) return
      data = await api.trackPlaylist(playlistInput.value.trim(), trustTier.value || null)
      lastIngestResult.value = { status: 'ingested', source_created: data.created, playlist: true }
      lastIngestJob.value = data.job || null
      playlistInput.value = ''
      return
    } else {
      if (!fileInput.value) return
      data = await api.ingestFile(fileInput.value, trustTier.value || null)
    }
    lastIngestResult.value = data.result
    lastIngestJob.value = data.job || null
    if (data.result?.status !== 'failed') {
      urlInput.value = ''
      youtubeInput.value = ''
      fileInput.value = null
      await loadSources()
    }
  } finally {
    submitting.value = false
  }
}

async function restoreSource(source) {
  await api.restoreSource(source.id)
  await loadSources()
}

function onFileChange(e) {
  fileInput.value = e.target.files[0] || null
}

const backfilling = ref(false)
const backfillResult = ref(null)
const sweepingAds = ref(false)
const adSweepJob = ref(null)
const playlists = ref([])
const playlistVideos = ref({})

async function runBackfillEmbeddings() {
  backfilling.value = true
  backfillResult.value = null
  try {
    const data = await api.backfillEmbeddings()
    backfillResult.value = data.result
  } finally {
    backfilling.value = false
  }
}

async function runAdSweep() {
  sweepingAds.value = true
  try {
    const data = await api.triggerAdSweep()
    adSweepJob.value = data.job
  } finally {
    sweepingAds.value = false
  }
}

async function runSearch() {
  if (!searchQuery.value.trim()) return
  searching.value = true
  try {
    const data = await api.searchChunks(searchQuery.value.trim(), searchSemantic.value, 15)
    searchResults.value = data.results || []
  } finally {
    searching.value = false
  }
}

function formatDate(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

const lifecycleLabels = {
  queued: 'Queued',
  running: 'Processing',
  ingested: 'Ingested',
  chunked: 'Chunked',
  ready: 'Ready',
  partial: 'Partial',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

const lifecycleClasses = {
  queued: 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300',
  running: 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300',
  ingested: 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300',
  chunked: 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300',
  ready: 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300',
  partial: 'bg-yellow-100 dark:bg-yellow-900/40 text-yellow-700 dark:text-yellow-300',
  failed: 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300',
  cancelled: 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300',
}
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-xl font-bold text-gray-900 dark:text-white">Sources</h2>
      <div class="flex items-center gap-2">
        <button @click="showArchived = !showArchived; loadSources()" class="px-3 py-1.5 text-sm rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700">
          {{ showArchived ? 'Hide archived' : 'Show archived' }}
        </button>
        <button
          @click="runBackfillEmbeddings"
          :disabled="backfilling"
          :title="backfillResult ? `chunks: ${backfillResult.chunks_embedded} embedded (${backfillResult.chunks_failed} failed), claims: ${backfillResult.claims_embedded} embedded (${backfillResult.claims_failed} failed)` : 'Embed any chunk/claim missing a vector'"
          class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 transition-colors"
        >
          <Layers class="w-4 h-4" />
          {{ backfilling ? 'Backfilling...' : 'Backfill embeddings' }}
        </button>
        <button
          @click="runAdSweep"
          :disabled="sweepingAds"
          title="Screen existing claims for confident sponsor/ad classifications"
          class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 transition-colors"
        >
          {{ sweepingAds ? 'Queueing...' : 'Screen ads' }}
        </button>
        <button
          @click="showSearch = !showSearch; showAdd = false"
          class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
        >
          <Search class="w-4 h-4" />
          Search content
        </button>
        <button
          @click="showAdd = !showAdd; showSearch = false"
          class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md bg-blue-600 hover:bg-blue-700 text-white transition-colors"
        >
          <Plus class="w-4 h-4" />
          Add Source
        </button>
      </div>
    </div>

    <!-- Search content -->
    <p v-if="adSweepJob" class="mb-3 text-xs text-blue-600 dark:text-blue-400">Ad screening is {{ adSweepJob.status }} and runs after active user work.</p>

    <div
      v-if="showSearch"
      class="mb-6 p-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg space-y-3"
    >
      <div class="flex gap-2">
        <input
          v-model="searchQuery"
          type="text"
          placeholder="Search chunked content..."
          @keydown.enter="runSearch"
          class="flex-1 px-3 py-2 text-sm rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
        />
        <label class="flex items-center gap-1.5 px-3 text-xs text-gray-500 dark:text-gray-400 cursor-pointer select-none">
          <input type="checkbox" v-model="searchSemantic" class="accent-blue-600" />
          Semantic
        </label>
        <button
          @click="runSearch"
          :disabled="searching || !searchQuery.trim()"
          class="px-3 py-1.5 text-sm rounded-md bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white transition-colors"
        >
          {{ searching ? 'Searching...' : 'Search' }}
        </button>
      </div>
      <div v-if="searchResults.length" class="space-y-2">
        <div
          v-for="(r, i) in searchResults"
          :key="i"
          class="text-sm border-l-2 border-gray-200 dark:border-gray-600 pl-3"
        >
          <p class="font-medium text-gray-900 dark:text-white">
            {{ r.source_title || r.canonical_uri }}
            <span v-if="searchSemantic" class="text-xs font-normal text-gray-400 dark:text-gray-500">score={{ r.score?.toFixed(3) }}</span>
          </p>
          <p class="text-xs text-gray-500 dark:text-gray-400" v-html="searchSemantic ? (r.chunk_text || '').slice(0, 300) : r.snippet"></p>
        </div>
      </div>
      <p v-else-if="!searching" class="text-xs text-gray-400 dark:text-gray-500">No results yet — try a search above.</p>
    </div>

    <!-- Add source -->
    <div
      v-if="showAdd"
      class="mb-6 p-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg space-y-3"
    >
      <div class="flex items-center gap-1 border-b border-gray-200 dark:border-gray-700">
        <button
          v-for="tab in [{ key: 'url', label: 'URL', icon: Globe }, { key: 'youtube', label: 'YouTube', icon: Youtube }, { key: 'playlist', label: 'Playlist', icon: Youtube }, { key: 'file', label: 'File', icon: FileUp }]"
          :key="tab.key"
          @click="addTab = tab.key"
          class="flex items-center gap-1.5 px-3 py-2 text-sm border-b-2 transition-colors"
          :class="addTab === tab.key
            ? 'border-blue-600 text-blue-600 dark:text-blue-400'
            : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'"
        >
          <component :is="tab.icon" class="w-3.5 h-3.5" />
          {{ tab.label }}
        </button>
      </div>

      <input
        v-if="addTab === 'url'"
        v-model="urlInput"
        type="text"
        placeholder="https://example.com/article"
        class="w-full px-3 py-2 text-sm rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
      />
      <input
        v-if="addTab === 'youtube'"
        v-model="youtubeInput"
        type="text"
        placeholder="YouTube URL or video ID"
        class="w-full px-3 py-2 text-sm rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
      />
      <input
        v-if="addTab === 'playlist'"
        v-model="playlistInput"
        type="text"
        placeholder="YouTube playlist URL — checks for new videos during idle time"
        class="w-full px-3 py-2 text-sm rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
      />
      <input
        v-if="addTab === 'file'"
        type="file"
        @change="onFileChange"
        accept=".pdf,.md,.txt,.html,.docx"
        class="w-full text-sm text-gray-700 dark:text-gray-300"
      />

      <div class="flex items-center gap-3">
        <select
          v-model="trustTier"
          class="px-3 py-1.5 text-sm rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
        >
          <option v-for="t in TRUST_TIERS" :key="t.value" :value="t.value">{{ t.label }}</option>
        </select>
        <button
          @click="submitIngest"
          :disabled="submitting"
          class="px-3 py-1.5 text-sm rounded-md bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white transition-colors"
        >
          {{ submitting ? 'Saving...' : (addTab === 'playlist' ? 'Track playlist' : 'Ingest') }}
        </button>
      </div>

      <div
        v-if="lastIngestResult"
        class="px-3 py-2 text-sm rounded-md"
        :class="lastIngestResult.status === 'failed'
          ? 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300'
          : 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300'"
      >
        <template v-if="lastIngestResult.status === 'failed'">Failed: {{ lastIngestResult.error }}</template>
        <template v-else-if="lastIngestResult.status === 'unchanged'">No change — content is identical to the latest version.</template>
        <template v-else-if="lastIngestResult.playlist">
          Playlist tracking is active. New videos will be added as normal sources during idle time.
          <span v-if="lastIngestJob"> Discovery is {{ lastIngestJob.status }}.</span>
        </template>
        <template v-else>
          Ingested a new version (source {{ lastIngestResult.source_created ? 'created' : 'existing' }}).
          <span v-if="lastIngestJob"> Automatic processing is {{ lastIngestJob.status }}.</span>
        </template>
      </div>
    </div>

    <div v-if="loading" class="text-sm text-gray-500 dark:text-gray-400">Loading...</div>

    <div v-else-if="sources.length === 0" class="text-center py-12 text-gray-500 dark:text-gray-400">
      <p class="text-sm">No sources ingested yet. Use "Add Source" above to get started.</p>
    </div>

    <div v-else class="space-y-2">
      <div
        v-for="source in sources"
        :key="source.id"
        @click="openSource(source)"
        class="flex items-center justify-between px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors"
      >
        <div class="min-w-0">
          <p class="text-sm font-medium text-gray-900 dark:text-white truncate">{{ source.title || source.canonical_uri }}</p>
          <p class="text-xs text-gray-500 dark:text-gray-400 truncate">{{ source.canonical_uri }}</p>
          <p class="text-xs mt-0.5">
            <span v-if="source.topic_names?.length" class="text-blue-600 dark:text-blue-400">
              {{ source.topic_names.join(', ') }}
            </span>
            <span
              class="ml-1.5"
              :class="source.claim_count > 0 ? 'text-gray-400 dark:text-gray-500' : 'text-gray-500 dark:text-gray-400'"
            >
              {{ source.claim_count > 0 ? `${source.claim_count} claim(s)` : 'no claims yet' }}
            </span>
            <span v-if="source.processing_error" class="ml-1.5 text-red-600 dark:text-red-400">{{ source.processing_error }}</span>
          </p>
        </div>
        <div class="flex items-center gap-2 shrink-0 ml-3 text-xs text-gray-400 dark:text-gray-500">
          <button v-if="!source.is_active" @click.stop="restoreSource(source)" class="text-blue-600 dark:text-blue-400 hover:underline">Restore</button>
          <span class="px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-700 uppercase">{{ source.source_type_code }}</span>
          <span class="px-1.5 py-0.5 rounded" :class="lifecycleClasses[source.lifecycle] || lifecycleClasses.ingested">
            {{ lifecycleLabels[source.lifecycle] || source.lifecycle }}
          </span>
          <span>{{ formatDate(source.updated_at) }}</span>
        </div>
      </div>
    </div>

    <details v-if="playlists.length" class="mt-6 text-sm">
      <summary class="cursor-pointer text-gray-600 dark:text-gray-300">Tracked playlists ({{ playlists.length }})</summary>
      <div v-for="playlist in playlists" :key="playlist.id" class="mt-2 p-3 border rounded border-gray-200 dark:border-gray-700">
        <div class="flex justify-between gap-2"><span>{{ playlist.title || playlist.url }}</span><span class="flex gap-2"><button @click="checkPlaylist(playlist)" class="text-blue-600 hover:underline">Check playlist</button><button @click="ingestPlaylistBatch(playlist)" class="text-blue-600 hover:underline">Ingest next batch</button><button @click="showPlaylistVideos(playlist)" class="text-blue-600 hover:underline">Show videos</button><button @click="removePlaylist(playlist)" class="text-red-600 hover:underline">Remove</button></span></div>
        <p v-for="video in playlistVideos[playlist.id] || []" :key="video.video_id" class="text-xs text-gray-500 mt-1">{{ video.title || video.video_id }} — {{ video.ingested_at ? 'ingested' : 'discovered' }}</p>
        <p v-if="playlistVideos[playlist.id]?.length === 0" class="text-xs text-gray-500 mt-1">No videos discovered yet. Check the playlist to start its first scan.</p>
      </div>
    </details>
  </div>
</template>
