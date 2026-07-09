<script setup>
import { ref, onMounted, computed } from 'vue'
import { Check, X, GitMerge } from 'lucide-vue-next'
import { useApi } from '../composables/useApi.js'

const api = useApi()

const TYPES = [
  { key: 'all', label: 'All', apiType: null },
  { key: 'entity_duplicate', label: 'Entities', apiType: 'entity_duplicate' },
  { key: 'claim_duplicate', label: 'Claims', apiType: 'claim_duplicate' },
  { key: 'claim_contradiction', label: 'Contradictions', apiType: 'claim_contradiction' },
]

const activeType = ref('all')
const candidates = ref([])
const loading = ref(true)
const reviewingId = ref(null)
const lastOutcome = ref(null)

// Guards against out-of-order responses: switching tabs quickly can let an
// earlier, slower request (e.g. "All", which fetches far more rows) resolve
// after a later, faster one (e.g. a small type filter) and clobber it with
// stale data. Only the response matching the request in flight when it
// resolves is applied.
let requestSeq = 0

onMounted(load)

async function load() {
  const thisRequest = ++requestSeq
  loading.value = true
  const type = TYPES.find(t => t.key === activeType.value)?.apiType
  const result = await api.fetchResolutionCandidates(type)
  if (thisRequest !== requestSeq) return
  candidates.value = result.candidates || []
  loading.value = false
}

function switchType(key) {
  activeType.value = key
  load()
}

function describeOutcome(result) {
  if (result.action === 'merged') {
    const noun = result.candidate_type === 'entity_duplicate' ? 'Entity' : 'Claim'
    return `${noun} ${result.loser_id.slice(0, 8)} merged into ${result.winner_id.slice(0, 8)}`
  }
  if (result.action === 'no_op_already_merged') {
    return 'Both sides already resolved to the same row — nothing to merge.'
  }
  if (result.action === 'contradiction_recorded') {
    return 'Contradiction recorded — both claims’ status updated, no merge performed.'
  }
  if (result.decision === 'rejected') {
    return 'Marked as rejected.'
  }
  return `Outcome: ${result.action}`
}

async function review(candidate, decision) {
  reviewingId.value = candidate.id
  try {
    const result = await api.reviewResolutionCandidate(candidate.id, decision)
    lastOutcome.value = describeOutcome(result)
    candidates.value = candidates.value.filter(c => c.id !== candidate.id)
  } finally {
    reviewingId.value = null
  }
}

const typeLabels = {
  entity_duplicate: 'Entity duplicate',
  claim_duplicate: 'Claim duplicate',
  claim_contradiction: 'Contradiction',
}

const typeBadgeColors = {
  entity_duplicate: 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300',
  claim_duplicate: 'bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300',
  claim_contradiction: 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300',
}

const isEmpty = computed(() => !loading.value && candidates.value.length === 0)
</script>

<template>
  <div>
    <div class="mb-4">
      <h2 class="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
        <GitMerge class="w-5 h-5" :stroke-width="1.5" />
        Resolution Queue
      </h2>
      <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
        Review entity/claim duplicates and confirmed contradictions found across the knowledge base.
        Accepting a duplicate merges it; nothing is ever auto-merged without review.
      </p>
    </div>

    <div
      v-if="lastOutcome"
      class="mb-4 px-4 py-2.5 text-sm rounded-md bg-green-50 dark:bg-green-900/20 text-green-800 dark:text-green-300 border border-green-200 dark:border-green-900/40"
    >
      {{ lastOutcome }}
    </div>

    <div class="flex items-center gap-1 mb-4 border-b border-gray-200 dark:border-gray-700">
      <button
        v-for="t in TYPES"
        :key="t.key"
        @click="switchType(t.key)"
        class="px-3 py-2 text-sm border-b-2 transition-colors"
        :class="activeType === t.key
          ? 'border-blue-600 text-blue-600 dark:text-blue-400'
          : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'"
      >
        {{ t.label }}
      </button>
    </div>

    <div v-if="loading" class="text-sm text-gray-500 dark:text-gray-400">Loading...</div>

    <div v-else-if="isEmpty" class="text-center py-12 text-gray-500 dark:text-gray-400">
      <p class="text-sm">No pending candidates in this category.</p>
    </div>

    <div v-else class="space-y-3">
      <div
        v-for="c in candidates"
        :key="c.id"
        class="px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg"
      >
        <div class="flex items-center justify-between gap-2 mb-2">
          <div class="flex items-center gap-2">
            <span
              class="px-1.5 py-0.5 text-[10px] rounded uppercase font-medium"
              :class="typeBadgeColors[c.candidate_type]"
            >
              {{ typeLabels[c.candidate_type] || c.candidate_type }}
            </span>
            <span class="text-xs text-gray-400 dark:text-gray-500">
              score={{ c.score?.toFixed(3) }} ({{ c.method }})
            </span>
          </div>
          <div class="flex items-center gap-1 shrink-0">
            <button
              @click="review(c, 'accepted')"
              :disabled="reviewingId === c.id"
              class="p-1.5 rounded-md hover:bg-green-50 dark:hover:bg-green-900/30 text-green-600 disabled:opacity-50"
              title="Accept"
            >
              <Check class="w-4 h-4" />
            </button>
            <button
              @click="review(c, 'rejected')"
              :disabled="reviewingId === c.id"
              class="p-1.5 rounded-md hover:bg-red-50 dark:hover:bg-red-900/30 text-red-500 disabled:opacity-50"
              title="Reject"
            >
              <X class="w-4 h-4" />
            </button>
          </div>
        </div>

        <div class="text-sm text-gray-900 dark:text-white space-y-1">
          <p><span class="text-gray-400 dark:text-gray-500">A:</span> {{ c.left_label }}</p>
          <p><span class="text-gray-400 dark:text-gray-500">B:</span> {{ c.right_label }}</p>
        </div>
        <p v-if="c.reason" class="text-xs text-gray-400 dark:text-gray-500 mt-1">{{ c.reason }}</p>
      </div>
    </div>
  </div>
</template>
