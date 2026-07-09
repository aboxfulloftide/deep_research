<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRoute } from 'vue-router'
import { marked } from 'marked'
import { Check, X, RefreshCw, FileText, Clock } from 'lucide-vue-next'
import { useApi } from '../composables/useApi.js'

const route = useRoute()
const api = useApi()

const topicId = computed(() => route.params.id)

const topic = ref(null)
const timeline = ref([])
const suggestedClaims = ref([])
const suggestedSources = ref([])
const report = ref(null)
const reportSuggestion = ref(null)
const activeTab = ref('timeline')
const loading = ref(true)
const backfilling = ref(false)
const generatingReport = ref(false)

const statusColors = {
  unverified: 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  supported: 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300',
  contradicted: 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300',
  mixed: 'bg-yellow-100 dark:bg-yellow-900/40 text-yellow-700 dark:text-yellow-300',
}

onMounted(loadAll)

async function loadAll() {
  loading.value = true
  const detail = await api.fetchTopic(topicId.value)
  topic.value = detail.topic

  const [tl, sc, ss, rep] = await Promise.all([
    api.fetchTimeline(topic.value.id),
    api.fetchTopicClaims(topic.value.id, 'suggested'),
    api.fetchTopicSources(topic.value.id, 'suggested'),
    api.fetchReport(topic.value.id),
  ])
  timeline.value = tl.entries || []
  suggestedClaims.value = sc.claims || []
  suggestedSources.value = ss.sources || []
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
</script>

<template>
  <div v-if="loading" class="text-sm text-gray-500 dark:text-gray-400">Loading...</div>

  <div v-else-if="topic">
    <div class="mb-4">
      <h2 class="text-xl font-bold text-gray-900 dark:text-white">{{ topic.name }}</h2>
      <p v-if="topic.description" class="text-sm text-gray-500 dark:text-gray-400 mt-1">
        {{ topic.description }}
      </p>
    </div>

    <div class="flex items-center gap-1 mb-4 border-b border-gray-200 dark:border-gray-700">
      <button
        v-for="tab in ['timeline', 'suggestions', 'report']"
        :key="tab"
        @click="activeTab = tab"
        class="px-3 py-2 text-sm capitalize border-b-2 transition-colors"
        :class="activeTab === tab
          ? 'border-blue-600 text-blue-600 dark:text-blue-400'
          : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'"
      >
        {{ tab }}
        <span
          v-if="tab === 'suggestions' && (suggestedClaims.length + suggestedSources.length) > 0"
          class="ml-1 px-1.5 py-0.5 text-xs rounded-full bg-yellow-100 dark:bg-yellow-900/40 text-yellow-700 dark:text-yellow-300"
        >
          {{ suggestedClaims.length + suggestedSources.length }}
        </span>
      </button>
    </div>

    <!-- Timeline -->
    <div v-if="activeTab === 'timeline'">
      <div v-if="timeline.length === 0" class="text-center py-12 text-gray-500 dark:text-gray-400">
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
