<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { ShieldCheck, PlayCircle, Activity } from 'lucide-vue-next'
import { useApi } from '../composables/useApi.js'

const api = useApi()

const currentRun = ref(null)
const runs = ref([])
const loading = ref(true)
const triggering = ref(false)
const triggerError = ref(null)
const jobs = ref([])

let pollHandle = null

onMounted(() => {
  load()
  pollHandle = setInterval(load, 5000)
})

onUnmounted(() => {
  if (pollHandle) clearInterval(pollHandle)
})

async function load() {
  const [currentData, runsData, jobsData] = await Promise.all([
    api.fetchCurrentVerificationRun(),
    api.fetchVerificationRuns(),
    api.fetchProcessingJobs(),
  ])
  currentRun.value = currentData.run
  runs.value = runsData.runs || []
  jobs.value = jobsData.jobs || []
  loading.value = false
}

async function triggerNow() {
  triggering.value = true
  triggerError.value = null
  try {
    await api.triggerVerificationRun()
    await load()
  } catch (e) {
    triggerError.value = e.message
  } finally {
    triggering.value = false
  }
}

const isRunning = computed(() => currentRun.value != null)
const activeJobs = computed(() => jobs.value.filter(j => ['queued', 'running'].includes(j.status)))
const recentJobs = computed(() => jobs.value.filter(j => !['queued', 'running'].includes(j.status)).slice(0, 10))
const queuedJobs = computed(() => activeJobs.value
  .filter(job => job.status === 'queued')
  .sort((a, b) => b.priority - a.priority || new Date(a.created_at) - new Date(b.created_at)))

const STATUS_FIELDS = [
  { key: 'supported_count', label: 'Supported', color: 'text-green-600 dark:text-green-400' },
  { key: 'contradicted_count', label: 'Contradicted', color: 'text-red-600 dark:text-red-400' },
  { key: 'mixed_count', label: 'Mixed', color: 'text-yellow-600 dark:text-yellow-400' },
  { key: 'unverified_count', label: 'Unverified', color: 'text-gray-500 dark:text-gray-400' },
  { key: 'skipped_count', label: 'Skipped', color: 'text-gray-400 dark:text-gray-500' },
  { key: 'failed_count', label: 'Failed', color: 'text-red-700 dark:text-red-500' },
]

function formatDate(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit' })
}

function duration(run) {
  if (!run.started_at) return ''
  const end = run.completed_at ? new Date(run.completed_at) : new Date()
  const ms = end - new Date(run.started_at)
  const mins = Math.round(ms / 60000)
  if (mins < 1) return '<1 min'
  if (mins < 60) return `${mins} min`
  return `${Math.floor(mins / 60)}h ${mins % 60}m`
}

const statusColors = {
  running: 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300',
  completed: 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300',
  failed: 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300',
}

function jobLabel(job) {
  return ({ source_pipeline: 'Source ingest', source_verify: 'Source verification', topic_verify: 'Topic verification', claim_verify: 'Claim verification', verification_sweep: 'Verification sweep', playlist_poll: 'Playlist ingest', ad_sweep: 'Ad screening', contradiction_triage: 'Contradiction triage', counter_evidence: 'Counter-evidence', topic_discovery: 'Topic discovery' })[job.job_type] || job.job_type
}

function formatProgress(job) {
  return Object.entries(job.progress || {}).map(([key, value]) => `${key}: ${value}`).join(' · ')
}

const STAGE_PROGRESS = {
  source_pipeline: { queued: 0, trust: 5, chunk: 20, extract: 45, ad_check: 65, attach: 72, verify: 85, report: 95, complete: 100 },
  source_verify: { queued: 0, verify: 55, complete: 100 },
  topic_verify: { queued: 0, verify: 55, report: 90, complete: 100 },
  claim_verify: { queued: 0, verify: 55, complete: 100 },
  verification_sweep: { queued: 0, verify: 55, complete: 100 },
  playlist_poll: { queued: 0, discover: 45, complete: 100 },
  ad_sweep: { queued: 0, ad_check: 55, complete: 100 },
  contradiction_triage: { queued: 0, triage: 55, complete: 100 },
  counter_evidence: { queued: 0, counter_evidence: 55, complete: 100 },
  topic_discovery: { queued: 0, discover: 55, complete: 100 },
  model_experiment: { waiting_for_idle: 0, gather_sources: 25, evaluate: 65, complete: 100 },
}

function jobProgressPercent(job) {
  return STAGE_PROGRESS[job.job_type]?.[job.stage] ?? (job.status === 'queued' ? 0 : 50)
}

function stageLabel(job) {
  return ({ trust: 'Assessing source', chunk: 'Preparing content', extract: 'Extracting claims', ad_check: 'Screening ads', attach: 'Connecting topic', verify: 'Checking evidence', report: 'Refreshing report', discover: 'Discovering videos', triage: 'Reviewing contradiction', counter_evidence: 'Finding counter-evidence', gather_sources: 'Gathering test sources', evaluate: 'Running model test', waiting_for_idle: 'Waiting for idle GPU', queued: 'Waiting in queue' })[job.stage] || job.stage
}

function queuePosition(job) {
  const index = queuedJobs.value.findIndex(candidate => candidate.id === job.id)
  return index >= 0 ? `${index + 1} of ${queuedJobs.value.length} queued` : ''
}

function jobElapsed(job) {
  return duration({ started_at: job.started_at || job.created_at, completed_at: null })
}
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
        <ShieldCheck class="w-5 h-5" />
        Verification Status
      </h2>
      <div class="text-right">
        <button
          @click="triggerNow"
          :disabled="triggering || isRunning"
          class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          <PlayCircle class="w-4 h-4" />
          {{ isRunning ? 'Running...' : triggering ? 'Starting...' : 'Verify All Now' }}
        </button>
        <p v-if="triggerError" class="text-xs text-red-500 dark:text-red-400 mt-1">{{ triggerError }}</p>
      </div>
    </div>

    <p class="text-sm text-gray-500 dark:text-gray-400 mb-6">
      A nightly sweep (11pm–7am) checks every unverified claim above the importance threshold
      against independent sources, KB-internal first and the live web if that's thin — so a claim
      being in the system doesn't mean it's been taken at face value. You can also kick off the
      same sweep manually here.
    </p>

    <div v-if="loading" class="text-sm text-gray-500 dark:text-gray-400">Loading...</div>

    <template v-else>
      <section class="mb-8">
        <h3 class="text-sm font-semibold text-gray-900 dark:text-white mb-2 flex items-center gap-1.5"><Activity class="w-4 h-4" /> All Deep Research Activity</h3>
        <div v-if="activeJobs.length" class="space-y-2">
          <div v-for="job in activeJobs" :key="job.id" class="p-3 bg-white dark:bg-gray-800 border border-blue-200 dark:border-blue-800 rounded-lg text-sm">
            <div class="flex justify-between gap-3"><span class="font-medium text-gray-900 dark:text-white">{{ jobLabel(job) }}</span><span class="px-1.5 py-0.5 text-[10px] rounded uppercase font-medium" :class="statusColors.running">{{ job.status }} · {{ job.stage }}</span></div>
            <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">{{ job.source_id ? `source ${job.source_id.slice(0, 8)}` : job.topic_id ? `topic ${job.topic_id.slice(0, 8)}` : job.subject_type }}</p>
            <div class="mt-2">
              <div class="flex justify-between gap-2 text-[11px] text-gray-500 dark:text-gray-400"><span>{{ stageLabel(job) }} <span v-if="job.status === 'running'">· {{ jobElapsed(job) }} elapsed</span></span><span v-if="job.status === 'queued'">{{ queuePosition(job) }}</span><span v-else>~{{ jobProgressPercent(job) }}% stage estimate</span></div>
              <div class="mt-1 h-1.5 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700"><div class="h-full bg-blue-600 transition-all" :style="{ width: `${jobProgressPercent(job)}%` }"></div></div>
            </div>
            <p v-if="formatProgress(job)" class="text-xs text-gray-500 dark:text-gray-400 mt-1">{{ formatProgress(job) }}</p>
          </div>
        </div>
        <p v-else class="text-sm text-gray-500 dark:text-gray-400">No background work is currently running or queued.</p>
        <details v-if="recentJobs.length" class="mt-3 text-xs">
          <summary class="cursor-pointer text-gray-500 dark:text-gray-400">Recent completed or failed work</summary>
          <div v-for="job in recentJobs" :key="job.id" class="mt-1 flex justify-between gap-2 text-gray-500 dark:text-gray-400"><span>{{ jobLabel(job) }} · {{ job.status }}</span><span v-if="job.error_message" class="text-red-600 dark:text-red-400 truncate" :title="job.error_message">{{ job.error_message }}</span></div>
        </details>
      </section>

      <!-- Current run -->
      <div v-if="currentRun" class="mb-8 p-4 bg-white dark:bg-gray-800 border border-blue-300 dark:border-blue-700 rounded-lg">
        <div class="flex items-center justify-between mb-2">
          <span class="px-2 py-0.5 text-xs rounded uppercase font-medium" :class="statusColors.running">
            in progress
          </span>
          <span class="text-xs text-gray-400 dark:text-gray-500">
            started {{ formatDate(currentRun.started_at) }} · {{ currentRun.trigger }}
          </span>
        </div>
        <p class="text-sm text-gray-900 dark:text-white mb-1">
          {{ currentRun.claims_processed }} / {{ currentRun.claims_total }} claim(s) processed
        </p>
        <div class="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden mb-2">
          <div
            class="h-full bg-blue-600 transition-all"
            :style="{ width: currentRun.claims_total ? `${(100 * currentRun.claims_processed / currentRun.claims_total).toFixed(1)}%` : '0%' }"
          ></div>
        </div>
        <div v-if="currentRun.current_claim_texts?.length" class="text-xs text-gray-500 dark:text-gray-400 space-y-0.5">
          <p class="font-medium">Currently checking ({{ currentRun.current_claim_texts.length }} in flight):</p>
          <p v-for="(text, i) in currentRun.current_claim_texts" :key="i" class="italic truncate">{{ text }}</p>
        </div>
      </div>
      <div v-else class="mb-8 text-sm text-gray-500 dark:text-gray-400">
        No run in progress.
      </div>

      <!-- History -->
      <h3 class="text-sm font-semibold text-gray-900 dark:text-white mb-2">Run History</h3>
      <div v-if="runs.length === 0" class="text-sm text-gray-500 dark:text-gray-400">
        No verification runs yet.
      </div>
      <div v-else class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-gray-400 dark:text-gray-500 text-xs">
              <th class="pb-1 pr-3">Started</th>
              <th class="pb-1 pr-3">Trigger</th>
              <th class="pb-1 pr-3">Status</th>
              <th class="pb-1 pr-3">Duration</th>
              <th class="pb-1 pr-3">Claims</th>
              <th class="pb-1">Outcome</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="run in runs" :key="run.id" class="border-t border-gray-100 dark:border-gray-800">
              <td class="py-1.5 pr-3 text-gray-500 dark:text-gray-400 whitespace-nowrap">{{ formatDate(run.started_at) }}</td>
              <td class="py-1.5 pr-3">{{ run.trigger }}</td>
              <td class="py-1.5 pr-3">
                <span class="px-1.5 py-0.5 text-[10px] rounded uppercase font-medium" :class="statusColors[run.status] || statusColors.completed">
                  {{ run.status }}
                </span>
              </td>
              <td class="py-1.5 pr-3 text-gray-500 dark:text-gray-400 whitespace-nowrap">{{ duration(run) }}</td>
              <td class="py-1.5 pr-3">{{ run.claims_processed }} / {{ run.claims_total }}</td>
              <td class="py-1.5">
                <span
                  v-for="f in STATUS_FIELDS"
                  :key="f.key"
                  v-show="run[f.key] > 0"
                  class="mr-2 text-xs"
                  :class="f.color"
                >
                  {{ f.label.toLowerCase() }}={{ run[f.key] }}
                </span>
                <span v-if="run.error_message" class="text-xs text-red-600 dark:text-red-400" :title="run.error_message">error</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>
  </div>
</template>
