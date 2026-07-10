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

const showAdd = ref(false)
const addTab = ref('url')
const urlInput = ref('')
const youtubeInput = ref('')
const fileInput = ref(null)
const trustTier = ref('')
const submitting = ref(false)
const lastIngestResult = ref(null)

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

onMounted(loadSources)

async function loadSources() {
  loading.value = true
  const data = await api.fetchSources(query.value)
  sources.value = data.sources || []
  loading.value = false
}

function openSource(source) {
  router.push({ name: 'source', params: { id: source.id } })
}

async function submitIngest() {
  submitting.value = true
  lastIngestResult.value = null
  try {
    let data
    if (addTab.value === 'url') {
      if (!urlInput.value.trim()) return
      data = await api.ingestUrl(urlInput.value.trim(), trustTier.value || null)
    } else if (addTab.value === 'youtube') {
      if (!youtubeInput.value.trim()) return
      data = await api.ingestYoutube(youtubeInput.value.trim(), trustTier.value || null)
    } else {
      if (!fileInput.value) return
      data = await api.ingestFile(fileInput.value, trustTier.value || null)
    }
    lastIngestResult.value = data.result
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

function onFileChange(e) {
  fileInput.value = e.target.files[0] || null
}

const backfilling = ref(false)
const backfillResult = ref(null)

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
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-xl font-bold text-gray-900 dark:text-white">Sources</h2>
      <div class="flex items-center gap-2">
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
          v-for="tab in [{ key: 'url', label: 'URL', icon: Globe }, { key: 'youtube', label: 'YouTube', icon: Youtube }, { key: 'file', label: 'File', icon: FileUp }]"
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
          {{ submitting ? 'Ingesting...' : 'Ingest' }}
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
        <template v-else>Ingested a new version (source {{ lastIngestResult.source_created ? 'created' : 'existing' }}).</template>
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
        </div>
        <div class="flex items-center gap-2 shrink-0 ml-3 text-xs text-gray-400 dark:text-gray-500">
          <span class="px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-700 uppercase">{{ source.source_type_code }}</span>
          <span>{{ formatDate(source.updated_at) }}</span>
        </div>
      </div>
    </div>
  </div>
</template>
