<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute } from 'vue-router'
import { Scissors, Sparkles, ShieldCheck, Loader2 } from 'lucide-vue-next'
import { useApi } from '../composables/useApi.js'

const route = useRoute()
const api = useApi()

const sourceId = computed(() => route.params.id)
const source = ref(null)
const versions = ref([])
const fetchAttempts = ref([])
const keyPoints = ref([])
const decisions = ref([])
const loading = ref(true)
const backgroundProcessing = ref(false)
const processingJob = ref(null)
const processingError = ref(null)

const chunking = ref(false)
const chunkResult = ref(null)
const extracting = ref(false)
const extractForce = ref(false)
const extractResult = ref(null)
const verifying = ref(false)
const verifyForce = ref(false)
const verifyThreshold = ref('')
const verifyResult = ref(null)

let processingPollHandle = null

onMounted(async () => {
  await load()
  processingPollHandle = setInterval(checkProcessingStatus, 3000)
  checkProcessingStatus()
})

onUnmounted(() => {
  if (processingPollHandle) clearInterval(processingPollHandle)
})

async function load() {
  loading.value = true
  const data = await api.fetchSource(sourceId.value)
  source.value = data.source
  versions.value = data.versions || []
  fetchAttempts.value = data.fetch_attempts || []
  loading.value = false
  await loadKeyPoints()
  const decisionData = await api.fetchSourceDecisions(sourceId.value)
  decisions.value = decisionData.decisions || []
}

async function loadKeyPoints() {
  const data = await api.fetchSourceClaims(sourceId.value)
  keyPoints.value = data.claims || []
}

async function checkProcessingStatus() {
  const wasProcessing = backgroundProcessing.value
  const data = await api.fetchSourceProcessingStatus(sourceId.value)
  backgroundProcessing.value = !!data.processing
  processingJob.value = data.job || null
  if (processingJob.value && ['failed', 'partial'].includes(processingJob.value.status)) {
    processingError.value = processingJob.value.error_message || 'Processing did not finish.'
  }
  // Just finished (e.g. a pasted conversation's chunk/extract/verify
  // background task) -- refresh so the new claims/versions actually show up
  // without the user having to manually reload the page.
  if (wasProcessing && !backgroundProcessing.value) {
    await load()
  }
}

async function cancelPipeline() {
  if (!processingJob.value) return
  const data = await api.cancelProcessingJob(processingJob.value.id)
  processingJob.value = data.job
}

async function retryPipeline() {
  if (!processingJob.value) return
  const data = await api.retryProcessingJob(processingJob.value.id)
  processingJob.value = data.job
  processingError.value = null
  backgroundProcessing.value = true
}

async function resetTrustTier() {
  const data = await api.resetSourceTrustTier(sourceId.value)
  source.value = data.source
  const decisionData = await api.fetchSourceDecisions(sourceId.value)
  decisions.value = decisionData.decisions || []
}

async function archiveSource() {
  const data = await api.archiveSource(sourceId.value)
  source.value = data.source
}

const keyPointStatusColors = {
  unverified: 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  supported: 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300',
  contradicted: 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300',
  mixed: 'bg-yellow-100 dark:bg-yellow-900/40 text-yellow-700 dark:text-yellow-300',
}

async function runChunk() {
  chunking.value = true
  chunkResult.value = null
  try {
    const data = await api.chunkSource(sourceId.value)
    chunkResult.value = { job: data.job }
    processingJob.value = data.job
    backgroundProcessing.value = true
  } finally {
    chunking.value = false
  }
}

async function runExtract() {
  extracting.value = true
  extractResult.value = null
  try {
    const data = await api.extractSource(sourceId.value, extractForce.value)
    extractResult.value = { job: data.job }
    processingJob.value = data.job
    backgroundProcessing.value = true
  } finally {
    extracting.value = false
  }
}

async function runVerify() {
  verifying.value = true
  verifyResult.value = null
  try {
    const threshold = verifyThreshold.value === '' ? null : Number(verifyThreshold.value)
    const data = await api.verifySource(sourceId.value, verifyForce.value, threshold)
    verifyResult.value = { job: data.job }
    processingJob.value = data.job
    backgroundProcessing.value = true
  } finally {
    verifying.value = false
  }
}

function formatDate(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit' })
}

const statusColors = {
  supported: 'text-green-600 dark:text-green-400',
  contradicted: 'text-red-600 dark:text-red-400',
  mixed: 'text-yellow-600 dark:text-yellow-400',
  unverified: 'text-gray-500 dark:text-gray-400',
}
</script>

<template>
  <div v-if="loading" class="text-sm text-gray-500 dark:text-gray-400">Loading...</div>

  <div v-else-if="source">
    <div class="mb-4">
      <h2 class="text-xl font-bold text-gray-900 dark:text-white">{{ source.title || source.canonical_uri }}</h2>
      <a :href="source.canonical_uri" target="_blank" rel="noopener" class="text-sm text-blue-600 dark:text-blue-400 hover:underline break-all">
        {{ source.canonical_uri }}
      </a>
      <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">
        id: {{ source.id }} · trust: {{ source.trust_tier_code || '(none)' }}
        <button v-if="source.trust_tier_code" @click="resetTrustTier" class="ml-2 hover:underline">reset to automatic</button>
        <button v-if="source.is_active" @click="archiveSource" class="ml-2 text-red-600 dark:text-red-400 hover:underline">archive</button>
      </p>
    </div>

    <div
      v-if="backgroundProcessing"
      class="flex items-center gap-2 mb-4 px-3 py-2 text-sm rounded-md bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300"
    >
      <Loader2 class="w-4 h-4 animate-spin shrink-0" />
      Processing in the background{{ processingJob?.stage ? `: ${processingJob.stage}` : '' }} — this can take a few minutes.
      Feel free to navigate away; it keeps running on the server and this page will refresh automatically when it's done.
      <button v-if="processingJob" @click="cancelPipeline" class="ml-auto text-xs underline">Cancel</button>
    </div>

    <div v-else-if="processingError" class="flex items-center gap-2 mb-4 px-3 py-2 text-sm rounded-md bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300">
      {{ processingError }}
      <button @click="retryPipeline" class="ml-auto text-xs underline">Retry</button>
    </div>

    <!-- Actions -->
    <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
      <div class="p-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
        <button
          @click="runChunk"
          :disabled="chunking"
          class="flex items-center gap-1.5 w-full justify-center px-3 py-1.5 text-sm rounded-md bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50 transition-colors"
        >
          <Scissors class="w-3.5 h-3.5" />
          {{ chunking ? 'Chunking...' : 'Chunk' }}
        </button>
        <p v-if="chunkResult" class="text-xs text-gray-500 dark:text-gray-400 mt-2">
          Processing queued.
        </p>
      </div>

      <div class="p-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
        <button
          @click="runExtract"
          :disabled="extracting"
          class="flex items-center gap-1.5 w-full justify-center px-3 py-1.5 text-sm rounded-md bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50 transition-colors"
        >
          <Sparkles class="w-3.5 h-3.5" />
          {{ extracting ? 'Extracting...' : 'Extract' }}
        </button>
        <label class="flex items-center gap-1.5 mt-1.5 text-xs text-gray-500 dark:text-gray-400 cursor-pointer select-none">
          <input type="checkbox" v-model="extractForce" class="accent-blue-600" />
          Force re-extract
        </label>
        <div v-if="extractResult" class="text-xs text-gray-500 dark:text-gray-400 mt-2">
          <p>Processing queued.</p>
        </div>
      </div>

      <div class="p-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
        <button
          @click="runVerify"
          :disabled="verifying"
          class="flex items-center gap-1.5 w-full justify-center px-3 py-1.5 text-sm rounded-md bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50 transition-colors"
        >
          <ShieldCheck class="w-3.5 h-3.5" />
          {{ verifying ? 'Verifying...' : 'Verify claims' }}
        </button>
        <div class="flex items-center gap-2 mt-1.5">
          <label class="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400 cursor-pointer select-none">
            <input type="checkbox" v-model="verifyForce" class="accent-blue-600" />
            Force
          </label>
          <input
            v-model="verifyThreshold"
            type="number" step="0.05" min="0" max="1" placeholder="threshold"
            class="w-20 px-1.5 py-0.5 text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900"
          />
        </div>
        <div v-if="verifyResult" class="text-xs text-gray-500 dark:text-gray-400 mt-2 space-y-1">
          <p>Verification queued — it will run after any active processing.</p>
        </div>
      </div>
    </div>

    <!-- Key Points -->
    <h3 class="text-sm font-semibold text-gray-900 dark:text-white mb-2">
      Key Points
      <span v-if="keyPoints.length" class="font-normal text-gray-400 dark:text-gray-500">({{ keyPoints.length }})</span>
    </h3>

    <div v-if="keyPoints.length === 0" class="mb-6 text-sm text-gray-500 dark:text-gray-400">
      No claims extracted from this source yet — chunk and extract it to populate this.
    </div>
    <div v-else class="mb-6 space-y-2">
      <div
        v-for="claim in keyPoints"
        :key="claim.id"
        class="flex items-start gap-2 px-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm"
      >
        <span
          class="mt-0.5 shrink-0 px-1.5 py-0.5 text-[10px] rounded uppercase font-medium"
          :class="keyPointStatusColors[claim.status] || keyPointStatusColors.unverified"
        >
          {{ claim.status }}
        </span>
        <span class="text-gray-900 dark:text-white">{{ claim.canonical_text }}</span>
      </div>
    </div>

    <details v-if="decisions.length" class="mb-6 text-sm">
      <summary class="cursor-pointer text-gray-700 dark:text-gray-300">Automation history ({{ decisions.length }})</summary>
      <div class="mt-2 space-y-2">
        <div v-for="decision in decisions" :key="decision.id" class="px-3 py-2 bg-gray-50 dark:bg-gray-800 rounded-md text-xs">
          <p class="font-medium text-gray-800 dark:text-gray-200">{{ decision.decision }}</p>
          <p v-if="decision.reasoning" class="mt-0.5 text-gray-500 dark:text-gray-400">{{ decision.reasoning }}</p>
          <p class="mt-0.5 text-gray-400 dark:text-gray-500">{{ formatDate(decision.decided_at) }}<template v-if="decision.confidence !== null"> · {{ Math.round(decision.confidence * 100) }}% confidence</template></p>
        </div>
      </div>
    </details>

    <!-- Versions -->
    <h3 class="text-sm font-semibold text-gray-900 dark:text-white mb-2">Versions</h3>
    <div class="mb-6 overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="text-left text-gray-400 dark:text-gray-500 text-xs">
            <th class="pb-1 pr-3">#</th>
            <th class="pb-1 pr-3">Captured</th>
            <th class="pb-1 pr-3">First</th>
            <th class="pb-1 pr-3">Latest</th>
            <th class="pb-1 pr-3">Locked</th>
            <th class="pb-1">Bytes</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="v in versions" :key="v.id" class="border-t border-gray-100 dark:border-gray-800">
            <td class="py-1 pr-3">{{ v.version_number }}</td>
            <td class="py-1 pr-3 text-gray-500 dark:text-gray-400">{{ formatDate(v.captured_at) }}</td>
            <td class="py-1 pr-3">{{ v.is_first_version ? 'yes' : '' }}</td>
            <td class="py-1 pr-3">{{ v.is_latest ? 'yes' : '' }}</td>
            <td class="py-1 pr-3">{{ v.retention_locked ? 'yes' : '' }}</td>
            <td class="py-1">{{ v.byte_size || '' }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Fetch attempts -->
    <h3 class="text-sm font-semibold text-gray-900 dark:text-white mb-2">Fetch Attempts</h3>
    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="text-left text-gray-400 dark:text-gray-500 text-xs">
            <th class="pb-1 pr-3">When</th>
            <th class="pb-1 pr-3">Type</th>
            <th class="pb-1 pr-3">Status</th>
            <th class="pb-1">Error</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="a in fetchAttempts.slice(0, 10)" :key="a.id" class="border-t border-gray-100 dark:border-gray-800">
            <td class="py-1 pr-3 text-gray-500 dark:text-gray-400">{{ formatDate(a.created_at) }}</td>
            <td class="py-1 pr-3">{{ a.attempt_type }}</td>
            <td class="py-1 pr-3">{{ a.status }}</td>
            <td class="py-1 text-red-500 dark:text-red-400">{{ a.error_message || '' }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
