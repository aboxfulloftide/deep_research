<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { marked } from 'marked'
import { Check, X, RefreshCw, FileText, Clock, Loader2, ShieldCheck, Link, Upload } from 'lucide-vue-next'
import ClaimListItem from '../components/ClaimListItem.vue'
import { useApi } from '../composables/useApi.js'

const route = useRoute()
const router = useRouter()
const api = useApi()

const topicId = computed(() => route.params.id)
const processingSourceId = computed(() => route.query.processingSource || null)

const topic = ref(null)
const timeline = ref([])
const transcript = ref([])
const attachedClaims = ref([])
const suggestedClaims = ref([])
const suggestedSources = ref([])
const attachedSources = ref([])
const report = ref(null)
const reportSuggestion = ref(null)
const activeTab = ref(processingSourceId.value ? 'claims' : 'timeline')
const loading = ref(true)
const backfilling = ref(false)
const generatingReport = ref(false)
const backgroundProcessing = ref(false)
const topicProcessing = ref(false)
const triggerError = ref(null)
const sourceUrl = ref('')
const addingSource = ref(false)
const intakeError = ref(null)
const processingJob = ref(null)

const anyProcessing = computed(() => backgroundProcessing.value || topicProcessing.value)

const statusColors = {
  unverified: 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  supported: 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300',
  contradicted: 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300',
  mixed: 'bg-yellow-100 dark:bg-yellow-900/40 text-yellow-700 dark:text-yellow-300',
}

let processingPollHandle = null
let topicPollHandle = null

onMounted(async () => {
  await loadAll()
  if (processingSourceId.value) {
    processingPollHandle = setInterval(checkProcessingStatus, 3000)
    checkProcessingStatus()
  }
  // A manual "check flagged claims now" trigger from an earlier visit to
  // this page may still be running -- pick its progress back up.
  const data = await api.fetchTopicProcessingStatus(topicId.value)
  if (data.processing) {
    topicProcessing.value = true
    topicPollHandle = setInterval(checkTopicProcessingStatus, 3000)
  }
})

onUnmounted(() => {
  if (processingPollHandle) clearInterval(processingPollHandle)
  if (topicPollHandle) clearInterval(topicPollHandle)
})

async function checkProcessingStatus() {
  const wasProcessing = backgroundProcessing.value
  const data = await api.fetchSourceProcessingStatus(processingSourceId.value)
  backgroundProcessing.value = !!data.processing
  processingJob.value = data.job || null
  if (processingJob.value && ['failed', 'partial'].includes(processingJob.value.status)) {
    intakeError.value = processingJob.value.error_message || 'Source processing did not finish.'
  }
  if (wasProcessing && !backgroundProcessing.value) {
    await loadAll()
  }
}

async function addUrlSource() {
  if (!sourceUrl.value.trim()) return
  addingSource.value = true
  intakeError.value = null
  try {
    const urls = sourceUrl.value.split(/\s+/).map(url => url.trim()).filter(Boolean)
    const results = await Promise.all(urls.map((url) => {
      const isYoutube = /(?:youtube\.com|youtu\.be)/i.test(url)
      return isYoutube
        ? api.ingestTopicYoutube(topic.value.id, url)
        : api.ingestTopicUrl(topic.value.id, url)
    }))
    const data = results.at(-1)
    sourceUrl.value = ''
    processingJob.value = data.job
    backgroundProcessing.value = true
    await router.replace({ query: { ...route.query, processingSource: data.result.source_id } })
    if (!processingPollHandle) processingPollHandle = setInterval(checkProcessingStatus, 3000)
    topicProcessing.value = true
    if (!topicPollHandle) topicPollHandle = setInterval(checkTopicProcessingStatus, 3000)
    checkProcessingStatus()
    checkTopicProcessingStatus()
    await loadAll()
  } catch (e) {
    intakeError.value = e.message
  } finally {
    addingSource.value = false
  }
}

async function addFileSource(event) {
  const file = event.target.files?.[0]
  if (!file) return
  addingSource.value = true
  intakeError.value = null
  try {
    const data = await api.ingestTopicFile(topic.value.id, file)
    processingJob.value = data.job
    backgroundProcessing.value = true
    await router.replace({ query: { ...route.query, processingSource: data.result.source_id } })
    if (!processingPollHandle) processingPollHandle = setInterval(checkProcessingStatus, 3000)
    topicProcessing.value = true
    if (!topicPollHandle) topicPollHandle = setInterval(checkTopicProcessingStatus, 3000)
    checkProcessingStatus()
    checkTopicProcessingStatus()
    await loadAll()
  } catch (e) {
    intakeError.value = e.message
  } finally {
    event.target.value = ''
    addingSource.value = false
  }
}

async function cancelPipeline() {
  if (!processingJob.value) return
  try {
    const data = await api.cancelProcessingJob(processingJob.value.id)
    processingJob.value = data.job
  } catch (e) {
    intakeError.value = e.message
  }
}

async function retryPipeline() {
  if (!processingJob.value) return
  try {
    const data = await api.retryProcessingJob(processingJob.value.id)
    processingJob.value = data.job
    backgroundProcessing.value = true
  } catch (e) {
    intakeError.value = e.message
  }
}

async function checkTopicProcessingStatus() {
  const wasProcessing = topicProcessing.value
  const data = await api.fetchTopicProcessingStatus(topicId.value)
  topicProcessing.value = !!data.processing
  if (wasProcessing && !topicProcessing.value) {
    await loadAll()
    clearInterval(topicPollHandle)
    topicPollHandle = null
  }
}

async function triggerVerifyNow() {
  triggerError.value = null
  try {
    await api.triggerTopicVerification(topicId.value)
    topicProcessing.value = true
    if (!topicPollHandle) topicPollHandle = setInterval(checkTopicProcessingStatus, 3000)
  } catch (e) {
    triggerError.value = e.message
  }
}

async function loadAll() {
  loading.value = true
  const detail = await api.fetchTopic(topicId.value)
  topic.value = detail.topic

  const [tl, tr, ac, sc, ss, attached, rep] = await Promise.all([
    api.fetchTimeline(topic.value.id),
    api.fetchConversationTranscript(topic.value.id),
    api.fetchTopicClaims(topic.value.id, 'attached'),
    api.fetchTopicClaims(topic.value.id, 'suggested'),
    api.fetchTopicSources(topic.value.id, 'suggested'),
    api.fetchTopicSources(topic.value.id, 'attached'),
    api.fetchReport(topic.value.id),
  ])
  timeline.value = tl.entries || []
  transcript.value = tr.turns || []
  attachedClaims.value = ac.claims || []
  suggestedClaims.value = sc.claims || []
  suggestedSources.value = ss.sources || []
  attachedSources.value = attached.sources || []
  report.value = rep.report || null
  loading.value = false
}

async function acceptClaim(claim) {
  await api.reviewClaimSuggestion(topic.value.id, claim.id, 'attached')
  suggestedClaims.value = suggestedClaims.value.filter(c => c.id !== claim.id)
}

async function rejectClaim(claim) {
  await api.reviewClaimSuggestion(topic.value.id, claim.id, 'rejected')
  suggestedClaims.value = suggestedClaims.value.filter(c => c.id !== claim.id)
}

async function acceptSource(source) {
  await api.reviewSourceSuggestion(topic.value.id, source.id, 'attached')
  suggestedSources.value = suggestedSources.value.filter(s => s.id !== source.id)
}

async function rejectSource(source) {
  await api.reviewSourceSuggestion(topic.value.id, source.id, 'rejected')
  suggestedSources.value = suggestedSources.value.filter(s => s.id !== source.id)
}

async function removeAttachedSource(source) {
  await api.reviewSourceSuggestion(topic.value.id, source.id, 'rejected')
  attachedSources.value = attachedSources.value.filter(s => s.id !== source.id)
}

async function runBackfill() {
  backfilling.value = true
  try {
    await api.backfillTopic(topic.value.id)
    await loadAll()
  } finally {
    backfilling.value = false
  }
}

async function runGenerateReport() {
  generatingReport.value = true
  try {
    const result = await api.generateReport(topic.value.id)
    report.value = { content_markdown: result.content_markdown }
    reportSuggestion.value = result.suggestion || null
    activeTab.value = 'report'
  } finally {
    generatingReport.value = false
  }
}

function renderMarkdown(text) {
  if (!text) return ''
  return marked.parse(text, { breaks: true })
}

function formatReportDate(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit' })
}
</script>

<template>
  <div v-if="loading" class="text-sm text-gray-500 dark:text-gray-400">Loading...</div>

  <div v-else-if="topic">
    <div class="mb-4">
      <h2 class="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
        {{ topic.name }}
        <span
          v-if="topic.topic_type === 'conversation'"
          class="px-1.5 py-0.5 text-[10px] rounded uppercase font-medium bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300"
        >
          Conversation
        </span>
      </h2>
      <p v-if="topic.description" class="text-sm text-gray-500 dark:text-gray-400 mt-1">
        {{ topic.description }}
      </p>
    </div>

    <div
      v-if="anyProcessing"
      class="flex items-center gap-2 mb-4 px-3 py-2 text-sm rounded-md bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300"
    >
      <Loader2 class="w-4 h-4 animate-spin shrink-0" />
      {{ backgroundProcessing
        ? `Processing in the background${processingJob?.stage ? `: ${processingJob.stage}` : ''}`
        : 'Processing topic sources in the background' }}
      — this can take a few minutes.
      Feel free to navigate away; it keeps running on the server and this page will refresh automatically when it's done.
      <button v-if="backgroundProcessing && processingJob" @click="cancelPipeline" class="ml-auto text-xs underline">Cancel</button>
    </div>

    <details class="mb-4 text-sm" :open="attachedSources.length > 0">
      <summary class="cursor-pointer text-gray-700 dark:text-gray-300">
        Attached sources ({{ attachedSources.length }})
      </summary>
      <div v-if="attachedSources.length" class="mt-2 space-y-1.5">
        <div v-for="source in attachedSources" :key="source.id" class="flex items-center gap-2 px-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md">
          <a :href="source.canonical_uri" target="_blank" rel="noopener" class="min-w-0 flex-1 truncate text-blue-600 dark:text-blue-400 hover:underline" @click.stop>{{ source.title || source.canonical_uri }}</a>
          <button @click="removeAttachedSource(source)" class="text-xs text-red-600 dark:text-red-400 hover:underline">Remove</button>
        </div>
      </div>
      <p v-else class="mt-2 text-xs text-gray-500 dark:text-gray-400">Add a source above to start the automatic pipeline.</p>
    </details>

    <div class="mb-5 p-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
      <p class="text-sm font-medium text-gray-900 dark:text-white mb-2">Add sources</p>
      <div class="flex flex-col sm:flex-row gap-2">
        <textarea
          v-model="sourceUrl" @keydown.ctrl.enter.prevent="addUrlSource" @keydown.meta.enter.prevent="addUrlSource"
          rows="2"
          placeholder="Paste one or more article/YouTube URLs (one per line)"
          class="flex-1 px-3 py-2 text-sm rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
        ></textarea>
        <button @click="addUrlSource" :disabled="addingSource || !sourceUrl.trim()" class="flex items-center justify-center gap-1.5 px-3 py-2 text-sm rounded-md bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white">
          <Link class="w-4 h-4" /> {{ addingSource ? 'Adding...' : 'Add URL' }}
        </button>
        <label class="flex items-center justify-center gap-1.5 px-3 py-2 text-sm rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer">
          <Upload class="w-4 h-4" /> Add file
          <input type="file" class="hidden" @change="addFileSource" :disabled="addingSource" />
        </label>
      </div>
      <p class="mt-2 text-xs text-gray-500 dark:text-gray-400">Sources are attached, extracted, checked, and included in the overview automatically.</p>
      <p v-if="intakeError" class="mt-2 text-xs text-red-600 dark:text-red-400">{{ intakeError }} <button v-if="processingJob?.status === 'failed' || processingJob?.status === 'partial'" @click="retryPipeline" class="underline">Retry</button></p>
    </div>

    <div class="flex items-center gap-1 mb-4 border-b border-gray-200 dark:border-gray-700">
      <button
        v-for="tab in ['timeline', 'claims', 'suggestions', 'report']"
        :key="tab"
        @click="activeTab = tab"
        class="px-3 py-2 text-sm capitalize border-b-2 transition-colors"
        :class="activeTab === tab
          ? 'border-blue-600 text-blue-600 dark:text-blue-400'
          : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'"
      >
        {{ tab }}
        <span
          v-if="tab === 'claims' && attachedClaims.length > 0"
          class="ml-1 px-1.5 py-0.5 text-xs rounded-full bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300"
        >
          {{ attachedClaims.length }}
        </span>
        <span
          v-if="tab === 'suggestions' && (suggestedClaims.length + suggestedSources.length) > 0"
          class="ml-1 px-1.5 py-0.5 text-xs rounded-full bg-yellow-100 dark:bg-yellow-900/40 text-yellow-700 dark:text-yellow-300"
        >
          {{ suggestedClaims.length + suggestedSources.length }}
        </span>
      </button>
    </div>

    <!-- Claims -->
    <div v-if="activeTab === 'claims'">
      <div class="flex items-center justify-between mb-3">
        <p v-if="triggerError" class="text-xs text-red-500 dark:text-red-400">{{ triggerError }}</p>
        <div class="ml-auto">
          <button
            @click="triggerVerifyNow"
            :disabled="topicProcessing"
            class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 transition-colors"
          >
            <ShieldCheck class="w-3.5 h-3.5" />
            {{ topicProcessing ? 'Checking...' : 'Check flagged claims now' }}
          </button>
        </div>
      </div>

      <div v-if="attachedClaims.length === 0" class="text-center py-12 text-gray-500 dark:text-gray-400">
        <p class="text-sm">No claims attached yet.</p>
      </div>
      <div v-else class="space-y-2">
        <ClaimListItem v-for="claim in attachedClaims" :key="claim.id" :claim="claim" />
      </div>
    </div>

    <!-- Timeline -->
    <div v-if="activeTab === 'timeline'">
      <!-- A pasted conversation rarely has dated events, but its actual
           back-and-forth is far more useful context than an empty dated
           timeline -- show each turn with the claims checked from it right
           underneath, instead of the generic event timeline below. -->
      <div v-if="transcript.length > 0" class="space-y-3">
        <div
          v-for="turn in transcript"
          :key="turn.chunk_id"
          class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-4 py-3"
        >
          <p v-if="turn.speaker" class="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1">{{ turn.speaker }}</p>
          <p class="text-sm text-gray-900 dark:text-white whitespace-pre-wrap">{{ turn.text }}</p>
          <div v-if="turn.claims.length" class="mt-2.5 pt-2.5 border-t border-gray-100 dark:border-gray-700 space-y-2">
            <ClaimListItem v-for="claim in turn.claims" :key="claim.id" :claim="claim" />
          </div>
        </div>
      </div>

      <div v-else-if="timeline.length === 0" class="text-center py-12 text-gray-500 dark:text-gray-400">
        <Clock class="w-10 h-10 mx-auto mb-3 opacity-50" :stroke-width="1.5" />
        <p class="text-sm">No dated events yet. Attach sources and extract claims to populate the timeline.</p>
      </div>
      <div v-else class="relative pl-6 space-y-6 border-l-2 border-gray-200 dark:border-gray-700">
        <div v-for="entry in timeline" :key="entry.event.id" class="relative">
          <div class="absolute -left-[1.65rem] top-1 w-3 h-3 rounded-full bg-blue-600" />
          <p class="text-xs font-mono text-gray-400 dark:text-gray-500">{{ entry.event.start_at }}</p>
          <p class="text-sm font-semibold text-gray-900 dark:text-white">{{ entry.event.title }}</p>
          <ul class="mt-1 space-y-1">
            <li
              v-for="claim in entry.claims"
              :key="claim.id"
              class="text-sm text-gray-700 dark:text-gray-300 flex items-start gap-2"
            >
              <span
                class="mt-0.5 shrink-0 px-1.5 py-0.5 text-[10px] rounded uppercase font-medium"
                :class="statusColors[claim.status] || statusColors.unverified"
              >
                {{ claim.status }}
              </span>
              <span>{{ claim.canonical_text }}</span>
            </li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Suggestions -->
    <div v-else-if="activeTab === 'suggestions'">
      <div class="flex justify-end mb-3">
        <button
          @click="runBackfill"
          :disabled="backfilling"
          class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 transition-colors"
        >
          <RefreshCw class="w-3.5 h-3.5" :class="{ 'animate-spin': backfilling }" />
          {{ backfilling ? 'Scanning...' : 'Re-scan KB for suggestions' }}
        </button>
      </div>

      <div v-if="suggestedClaims.length === 0 && suggestedSources.length === 0" class="text-center py-12 text-gray-500 dark:text-gray-400">
        <p class="text-sm">No pending suggestions.</p>
      </div>

      <div v-else class="space-y-4">
        <div v-if="suggestedSources.length" class="space-y-2">
          <h3 class="text-sm font-semibold text-gray-900 dark:text-white">Sources</h3>
          <div
            v-for="source in suggestedSources"
            :key="source.id"
            class="flex items-center justify-between px-4 py-2.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg"
          >
            <div class="min-w-0">
              <p class="text-sm text-gray-900 dark:text-white truncate">{{ source.title || source.canonical_uri }}</p>
              <p class="text-xs text-gray-400 dark:text-gray-500">{{ source.link_reason }}</p>
            </div>
            <div class="flex items-center gap-1 shrink-0 ml-2">
              <button @click="acceptSource(source)" class="p-1.5 rounded-md hover:bg-green-50 dark:hover:bg-green-900/30 text-green-600" title="Accept">
                <Check class="w-4 h-4" />
              </button>
              <button @click="rejectSource(source)" class="p-1.5 rounded-md hover:bg-red-50 dark:hover:bg-red-900/30 text-red-500" title="Reject">
                <X class="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        <div v-if="suggestedClaims.length" class="space-y-2">
          <h3 class="text-sm font-semibold text-gray-900 dark:text-white">Claims</h3>
          <div
            v-for="claim in suggestedClaims"
            :key="claim.id"
            class="flex items-center justify-between px-4 py-2.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg"
          >
            <div class="min-w-0">
              <p class="text-sm text-gray-900 dark:text-white">{{ claim.canonical_text }}</p>
              <p class="text-xs text-gray-400 dark:text-gray-500">{{ claim.link_reason }}</p>
            </div>
            <div class="flex items-center gap-1 shrink-0 ml-2">
              <button @click="acceptClaim(claim)" class="p-1.5 rounded-md hover:bg-green-50 dark:hover:bg-green-900/30 text-green-600" title="Accept">
                <Check class="w-4 h-4" />
              </button>
              <button @click="rejectClaim(claim)" class="p-1.5 rounded-md hover:bg-red-50 dark:hover:bg-red-900/30 text-red-500" title="Reject">
                <X class="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Report -->
    <div v-else-if="activeTab === 'report'">
      <div class="flex justify-end mb-3">
        <button
          @click="runGenerateReport"
          :disabled="generatingReport"
          class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white transition-colors"
        >
          <FileText class="w-3.5 h-3.5" />
          {{ generatingReport ? 'Generating...' : (report ? 'Regenerate report' : 'Generate report') }}
        </button>
      </div>

      <div v-if="!report" class="text-center py-12 text-gray-500 dark:text-gray-400">
        <p class="text-sm">No report yet. Generate one from the current timeline and claims.</p>
      </div>
      <template v-else>
        <p class="mb-2 text-xs text-gray-400 dark:text-gray-500">
          Generated {{ formatReportDate(report.created_at) }} from {{ report.generated_from_scope?.claim_count ?? 0 }} attached claim(s).
        </p>
        <div
          v-if="reportSuggestion"
          class="mb-3 px-4 py-2.5 text-sm rounded-md bg-yellow-50 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-300 border border-yellow-200 dark:border-yellow-900/40"
        >
          {{ reportSuggestion }}
        </div>
        <div
          class="markdown-content text-sm bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-5"
          v-html="renderMarkdown(report.content_markdown)"
        />
      </template>
    </div>
  </div>
</template>
