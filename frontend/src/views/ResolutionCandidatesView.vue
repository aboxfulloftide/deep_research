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
const selectedIds = ref(new Set())
const bulkProcessing = ref(false)

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
  selectedIds.value = new Set()
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

const allSelected = computed(() =>
  candidates.value.length > 0 && candidates.value.every(c => selectedIds.value.has(c.id))
)

function toggleSelectAll() {
  selectedIds.value = allSelected.value
    ? new Set()
    : new Set(candidates.value.map(c => c.id))
}

function toggleSelect(id) {
  const next = new Set(selectedIds.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  selectedIds.value = next
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

function summarizeBulkOutcome(decision, ids, outcomes) {
  if (decision === 'rejected') {
    return `Rejected ${ids.length} candidate(s).`
  }
  const parts = []
  if (outcomes.merged) parts.push(`${outcomes.merged} merged`)
  if (outcomes.no_op_already_merged) parts.push(`${outcomes.no_op_already_merged} already resolved`)
  if (outcomes.contradiction_recorded) parts.push(`${outcomes.contradiction_recorded} contradiction(s) recorded`)
  if (outcomes.failed) parts.push(`${outcomes.failed} failed`)
  return `Accepted ${ids.length}: ${parts.join(', ')}.`
}

async function bulkReview(decision) {
  const ids = [...selectedIds.value]
  if (ids.length === 0 || bulkProcessing.value) return
  if (decision === 'accepted') {
    const ok = window.confirm(
      `Accept ${ids.length} candidate(s)? Duplicates will be merged and contradictions recorded — see ` +
      `"What does Accept actually do?" above for exactly what that means. This can't be bulk-undone.`
    )
    if (!ok) return
  }

  bulkProcessing.value = true
  const outcomes = {}
  try {
    for (const id of ids) {
      try {
        const result = await api.reviewResolutionCandidate(id, decision)
        outcomes[result.action] = (outcomes[result.action] || 0) + 1
      } catch (e) {
        outcomes.failed = (outcomes.failed || 0) + 1
      }
    }
    candidates.value = candidates.value.filter(c => !selectedIds.value.has(c.id))
    selectedIds.value = new Set()
    lastOutcome.value = summarizeBulkOutcome(decision, ids, outcomes)
  } finally {
    bulkProcessing.value = false
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

// What accepting/rejecting actually means differs by type -- these are the
// questions the score/evidence below can't answer for you.
const GUIDANCE = {
  all: 'Showing every pending item below. Pick a specific tab for guidance on what that type is asking you to judge.',
  entity_duplicate: 'Are A and B literally the same real-world thing, just named or spelled differently (e.g. "data center" vs. "data centers", a nickname vs. the full name)? Accept merges them into one entity from now on. Reject if they just look similar but are actually different things — nothing happens to either one.',
  claim_duplicate: 'Do A and B state the exact same fact, just worded differently? A high score here means the *wording* is similar, not that the facts match — check the evidence excerpts below for the actual numbers/dates/names. Claims that disagree on any of those are NOT duplicates even when the sentence structure looks alike, and both can be true at once (e.g. bank failure counts for two different years). Accept merges them, keeping the more important/reliable one and combining their evidence. Reject leaves both claims exactly as they are.',
  claim_contradiction: 'The system flagged A and B as possibly conflicting (one says something happened, the other says it didn’t, or gives an incompatible account). Accept only if they truly disagree about the same thing — this marks both claims’ status as disputed, it does not merge them, since a contradiction means two competing accounts, not one fact. Reject if they’re actually compatible (different time periods, one more specific than the other, etc.) — nothing changes for either claim.',
}
const currentGuidance = computed(() => GUIDANCE[activeType.value])

// The mechanics behind Accept/Reject -- what actually happens in the
// database, not just what question you're being asked to judge.
const MECHANICS = {
  entity_duplicate: 'Accept picks a winner (the older of the two entities) and moves everything that pointed at the loser — its metrics, any other pending candidates that reference it — onto the winner. The loser is then tombstoned: marked as merged (never deleted, so the record stays for audit), removed from every listing, and any future mention of that name resolves straight to the winner instead of creating a third duplicate. Reject just marks this pair reviewed and stops it reappearing — neither entity is touched.',
  claim_duplicate: 'Accept picks a winner (the claim with the higher importance score) and moves everything from the loser onto it — its evidence citations, any metrics, topic links — then tombstones the loser (marked deprecated, never deleted, so it stays for audit but disappears from every listing). This is the actual point of merging: two claims each backed by one source become one claim backed by two independent sources, which is what verification and trust scoring care about, and it stops a topic report from repeating the same fact twice in slightly different words. If that same fact gets extracted again later from a new source, it resolves straight to the winner instead of creating a third duplicate. Reject just marks this pair reviewed — both claims stay completely untouched and independent.',
  claim_contradiction: 'Accept updates both claims’ status to reflect the conflict (contradicted, or mixed if one side already had independent support) — nothing is merged, since a contradiction means two competing accounts, not the same fact. Reject just marks this pair reviewed — neither claim’s status changes.',
}
const currentMechanics = computed(() => MECHANICS[activeType.value])

const methodExplanations = {
  embedding_cosine: 'flagged by semantic similarity (meaning), not an exact text match',
  trigram: 'flagged by fuzzy name similarity (close spelling)',
  substring: 'flagged because one name contains the other',
}
</script>

<template>
  <div>
    <div class="mb-4">
      <h2 class="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
        <GitMerge class="w-5 h-5" :stroke-width="1.5" />
        Resolution Queue
      </h2>
      <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
        Everything here was found automatically and needs your judgment before anything happens —
        nothing is ever merged or flagged without a human decision. When in doubt, reject: it's a safe
        no-op that leaves every claim/entity exactly as it was.
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

    <div
      v-if="currentGuidance"
      class="mb-4 px-4 py-2.5 text-sm rounded-md bg-blue-50 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300 border border-blue-200 dark:border-blue-900/40"
    >
      <p>{{ currentGuidance }}</p>
      <details v-if="currentMechanics" class="mt-1.5">
        <summary class="cursor-pointer select-none text-xs font-medium hover:underline">
          What does Accept actually do?
        </summary>
        <p class="mt-1.5 text-xs text-blue-800/90 dark:text-blue-300/90">{{ currentMechanics }}</p>
      </details>
    </div>

    <div v-if="loading" class="text-sm text-gray-500 dark:text-gray-400">Loading...</div>

    <div v-else-if="isEmpty" class="text-center py-12 text-gray-500 dark:text-gray-400">
      <p class="text-sm">No pending candidates in this category.</p>
    </div>

    <template v-else>
      <div class="flex items-center gap-3 mb-3 text-sm">
        <label class="flex items-center gap-1.5 cursor-pointer select-none text-gray-600 dark:text-gray-300">
          <input type="checkbox" :checked="allSelected" @change="toggleSelectAll" class="accent-blue-600" />
          {{ allSelected ? 'Select none' : 'Select all' }}
        </label>
        <span v-if="selectedIds.size" class="text-gray-400 dark:text-gray-500">{{ selectedIds.size }} selected</span>
        <div v-if="selectedIds.size" class="flex items-center gap-2 ml-auto">
          <button
            @click="bulkReview('accepted')"
            :disabled="bulkProcessing"
            class="px-3 py-1.5 rounded-md bg-green-600 hover:bg-green-700 text-white text-xs disabled:opacity-50"
          >
            {{ bulkProcessing ? 'Working…' : `Accept selected (${selectedIds.size})` }}
          </button>
          <button
            @click="bulkReview('rejected')"
            :disabled="bulkProcessing"
            class="px-3 py-1.5 rounded-md bg-red-500 hover:bg-red-600 text-white text-xs disabled:opacity-50"
          >
            {{ bulkProcessing ? 'Working…' : `Reject selected (${selectedIds.size})` }}
          </button>
        </div>
      </div>

      <div class="space-y-3">
      <div
        v-for="c in candidates"
        :key="c.id"
        class="px-4 py-3 bg-white dark:bg-gray-800 border rounded-lg"
        :class="selectedIds.has(c.id)
          ? 'border-blue-400 dark:border-blue-600 ring-1 ring-blue-200 dark:ring-blue-900'
          : 'border-gray-200 dark:border-gray-700'"
      >
        <div class="flex items-center justify-between gap-2 mb-2">
          <div class="flex items-center gap-2">
            <input
              type="checkbox"
              :checked="selectedIds.has(c.id)"
              @change="toggleSelect(c.id)"
              class="accent-blue-600"
            />
            <span
              class="px-1.5 py-0.5 text-[10px] rounded uppercase font-medium"
              :class="typeBadgeColors[c.candidate_type]"
            >
              {{ typeLabels[c.candidate_type] || c.candidate_type }}
            </span>
            <span
              class="text-xs text-gray-400 dark:text-gray-500"
              :title="methodExplanations[c.method] || ''"
            >
              score={{ c.score?.toFixed(3) }} ({{ c.method }})
            </span>
          </div>
          <div class="flex items-center gap-1 shrink-0">
            <button
              @click="review(c, 'accepted')"
              :disabled="reviewingId === c.id"
              class="p-1.5 rounded-md hover:bg-green-50 dark:hover:bg-green-900/30 text-green-600 disabled:opacity-50"
              :title="c.candidate_type === 'claim_contradiction'
                ? 'Accept — record as a genuine contradiction (no merge)'
                : 'Accept — merge A and B into one'"
            >
              <Check class="w-4 h-4" />
            </button>
            <button
              @click="review(c, 'rejected')"
              :disabled="reviewingId === c.id"
              class="p-1.5 rounded-md hover:bg-red-50 dark:hover:bg-red-900/30 text-red-500 disabled:opacity-50"
              title="Reject — leave both exactly as they are"
            >
              <X class="w-4 h-4" />
            </button>
          </div>
        </div>

        <div class="text-sm text-gray-900 dark:text-white space-y-2.5">
          <div v-for="side in [{ label: 'A', text: c.left_label, type: c.left_entity_type, evidence: c.left_evidence },
                                 { label: 'B', text: c.right_label, type: c.right_entity_type, evidence: c.right_evidence }]"
               :key="side.label">
            <p>
              <span class="text-gray-400 dark:text-gray-500">{{ side.label }}:</span>
              {{ side.text }}
              <span v-if="side.type" class="text-xs text-gray-400 dark:text-gray-500">({{ side.type }})</span>
            </p>
            <div v-if="side.evidence?.length" class="mt-1 ml-4 space-y-1">
              <div
                v-for="(ev, i) in side.evidence"
                :key="i"
                class="text-xs border-l-2 border-gray-200 dark:border-gray-600 pl-2"
              >
                <span class="font-medium text-gray-500 dark:text-gray-400">{{ ev.source_title }}</span>
                <span v-if="ev.excerpt" class="text-gray-500 dark:text-gray-400 italic"> — "{{ ev.excerpt }}"</span>
              </div>
            </div>
          </div>
        </div>
        <p v-if="c.reason" class="text-xs text-gray-400 dark:text-gray-500 mt-1">{{ c.reason }}</p>
      </div>
      </div>
    </template>
  </div>
</template>
