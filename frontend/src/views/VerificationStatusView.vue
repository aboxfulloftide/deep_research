<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { ShieldCheck, PlayCircle } from 'lucide-vue-next'
import { useApi } from '../composables/useApi.js'

const api = useApi()

const currentRun = ref(null)
const runs = ref([])
const loading = ref(true)
const triggering = ref(false)
const triggerError = ref(null)

let pollHandle = null

onMounted(() => {
  load()
  pollHandle = setInterval(load, 5000)
})

onUnmounted(() => {
  if (pollHandle) clearInterval(pollHandle)
})

async function load() {
  const [currentData, runsData] = await Promise.all([
    api.fetchCurrentVerificationRun(),
    api.fetchVerificationRuns(),
  ])
  currentRun.value = currentData.run
  runs.value = runsData.runs || []
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
