<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ShieldCheck, ChevronDown, ChevronRight, Star, ThumbsUp, ThumbsDown, Flag } from 'lucide-vue-next'
import { useApi } from '../composables/useApi.js'

const props = defineProps({
  claim: { type: Object, required: true },
})

const api = useApi()
const router = useRouter()

// Local reactive copy so this card can update its own badges/status after an
// action (verify, flag toggle, contradiction review) without the parent list
// needing to know or re-fetch -- each card is self-contained.
const localClaim = ref({ ...props.claim })

const expanded = ref(false)
const detail = ref(null)
const decisions = ref([])
const verifying = ref(false)
const verifyForce = ref(false)
const reviewingCandidateId = ref(null)
const togglingCheck = ref(false)
const contextDraft = ref(props.claim.verification_context || '')
const savingContext = ref(false)
const editingContext = ref(false)
const findingCounter = ref(false)

const statusColors = {
  unverified: 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  supported: 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300',
  contradicted: 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300',
  mixed: 'bg-yellow-100 dark:bg-yellow-900/40 text-yellow-700 dark:text-yellow-300',
  deprecated: 'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 line-through',
}

// Not every extracted statement needs a claim check -- check_status explains
// the system's automatic call (based on importance) or a manual override.
// 'checked'/'checked_pending_retry' cover claims already attempted: a
// settled verdict (supported/contradicted/mixed) is done for good, but an
// inconclusive first pass ("unverified") gets another look automatically
// once UNVERIFIED_RETRY_COOLDOWN_HOURS has passed -- see claim_check_status.
const CHECK_STATUS_LABELS = {
  auto_check: 'Will check',
  auto_skip: "Won't check (low importance)",
  manual_include: 'Flagged to check',
  manual_exclude: 'Excluded from check',
  checked: 'Checked',
  checked_pending_retry: 'Inconclusive — retry pending',
  deprecated: 'Deprecated (merged)',
}
const CHECK_STATUS_ON = new Set(['auto_check', 'manual_include'])
// Flagging/deflagging only affects claims that haven't locked in an attempt
// yet -- once verification_attempted_at is set, the Flag override has no
// effect on a settled claim (never rechecked) or a claim still inside its
// retry cooldown (see is_claim_eligible_for_verification), so the button
// would just make a promise the nightly sweep won't keep. Same for a
// deprecated (merged-away) claim -- it's never verified at all. Use the
// "Verify" button below for an immediate, forced check instead.
const CHECK_STATUS_LOCKED = new Set(['checked', 'checked_pending_retry', 'deprecated'])

function willCheck() {
  return CHECK_STATUS_ON.has(localClaim.value.check_status)
}

function isLocked() {
  return CHECK_STATUS_LOCKED.has(localClaim.value.check_status)
}

async function toggleExpand() {
  expanded.value = !expanded.value
  if (expanded.value && !detail.value) {
    const [claimDetail, decisionData] = await Promise.all([
      api.fetchClaim(localClaim.value.id), api.fetchClaimDecisions(localClaim.value.id),
    ])
    detail.value = claimDetail
    decisions.value = decisionData.decisions || []
  }
}

async function runVerify() {
  verifying.value = true
  try {
    await api.verifyClaim(localClaim.value.id, verifyForce.value)
    // Verification is queued so it shares the global worker with source
    // processing. Refresh the expanded detail on the next visit/poll rather
    // than pretending an immediate verdict exists.
    detail.value = null
  } finally {
    verifying.value = false
  }
}

async function makePreferred(sourceId) {
  await api.setPreferredSource(localClaim.value.id, sourceId)
  detail.value = await api.fetchClaim(localClaim.value.id)
}

async function saveContext() {
  savingContext.value = true
  try {
    const data = await api.setClaimVerificationContext(localClaim.value.id, contextDraft.value.trim() || null)
    localClaim.value = { ...localClaim.value, ...data.claim }
    editingContext.value = false
  } finally {
    savingContext.value = false
  }
}

async function reviewContradiction(rc, decision) {
  reviewingCandidateId.value = rc.candidate_id
  try {
    await api.reviewResolutionCandidate(rc.candidate_id, decision)
    detail.value = await api.fetchClaim(localClaim.value.id)
    localClaim.value = { ...localClaim.value, status: detail.value.claim.status }
  } finally {
    reviewingCandidateId.value = null
  }
}

async function toggleCheckOverride() {
  togglingCheck.value = true
  try {
    const data = await api.setClaimVerificationOverride(localClaim.value.id, willCheck() ? 'exclude' : 'include')
    localClaim.value = { ...localClaim.value, ...data.claim }
  } finally {
    togglingCheck.value = false
  }
}

async function resetCheckOverride() {
  togglingCheck.value = true
  try {
    const data = await api.setClaimVerificationOverride(localClaim.value.id, null)
    localClaim.value = { ...localClaim.value, ...data.claim }
  } finally {
    togglingCheck.value = false
  }
}

async function runCounterEvidence() {
  findingCounter.value = true
  try {
    await api.findCounterEvidence(localClaim.value.id)
    detail.value = null
  } finally {
    findingCounter.value = false
  }
}
</script>

<template>
  <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
    <div
      @click="toggleExpand"
      class="flex items-start gap-2 px-4 py-3 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
    >
      <component :is="expanded ? ChevronDown : ChevronRight" class="w-4 h-4 mt-0.5 shrink-0 text-gray-400" />
      <div class="min-w-0 flex-1">
        <p class="text-sm text-gray-900 dark:text-white">{{ localClaim.canonical_text }}</p>
        <div class="flex items-center gap-2 mt-1">
          <span class="px-1.5 py-0.5 text-[10px] rounded uppercase font-medium" :class="statusColors[localClaim.status] || statusColors.unverified">
            {{ localClaim.status }}
          </span>
          <span v-if="localClaim.claim_type" class="text-xs text-gray-400 dark:text-gray-500">{{ localClaim.claim_type }}</span>
          <span v-if="localClaim.importance_score != null" class="text-xs text-gray-400 dark:text-gray-500">
            importance={{ localClaim.importance_score.toFixed(2) }}
          </span>
          <template v-if="localClaim.topics?.length">
            <a
              v-for="t in localClaim.topics"
              :key="t.id"
              href="#"
              @click.stop.prevent="router.push({ name: 'topic', params: { id: t.id } })"
              class="text-xs text-blue-600 dark:text-blue-400 hover:underline"
            >
              {{ t.name }}
            </a>
          </template>
          <!-- localClaim.topics is only present on list endpoints that
               enrich it (e.g. the general Claims page) -- if this claim is
               already being shown scoped to one topic (e.g. a topic's own
               Claims tab), the field is absent and there's nothing useful
               to add here. -->
          <span v-else-if="localClaim.topics" class="text-xs text-yellow-600 dark:text-yellow-400">not tied to a topic</span>
        </div>
      </div>
      <div v-if="localClaim.check_status" class="flex items-center gap-1.5 shrink-0" @click.stop>
        <span class="text-xs" :class="willCheck() ? 'text-blue-600 dark:text-blue-400' : 'text-gray-400 dark:text-gray-500'">
          {{ CHECK_STATUS_LABELS[localClaim.check_status] }}
        </span>
        <button
          v-if="!isLocked()"
          @click="toggleCheckOverride"
          :disabled="togglingCheck"
          class="flex items-center gap-1 px-2 py-1 text-xs rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 transition-colors"
          :title="willCheck() ? 'Deflag: exclude from auto-check' : 'Flag: include in auto-check'"
        >
          <Flag class="w-3 h-3" :class="{ 'fill-current': willCheck() }" />
          {{ willCheck() ? 'Deflag' : 'Flag' }}
        </button>
        <button
          v-if="localClaim.verification_override && !isLocked()"
          @click="resetCheckOverride"
          class="text-xs text-gray-400 dark:text-gray-500 hover:underline"
          title="Reset to automatic decision"
        >
          reset
        </button>
      </div>
    </div>

    <div v-if="expanded" class="px-4 pb-3 border-t border-gray-100 dark:border-gray-700 pt-3">
      <div v-if="!detail" class="text-xs text-gray-400 dark:text-gray-500">Loading evidence...</div>
      <template v-else>
        <div class="mb-3">
          <div v-if="!editingContext" class="flex items-start gap-2">
            <p v-if="localClaim.verification_context" class="text-xs text-gray-600 dark:text-gray-300 italic flex-1">
              Verification context: {{ localClaim.verification_context }}
            </p>
            <p v-else class="text-xs text-gray-400 dark:text-gray-500 flex-1">
              No added verification context.
            </p>
            <button
              @click="editingContext = true"
              class="text-xs text-blue-600 dark:text-blue-400 hover:underline shrink-0"
            >
              {{ localClaim.verification_context ? 'edit' : 'add context' }}
            </button>
          </div>
          <div v-else class="space-y-1.5">
            <textarea
              v-model="contextDraft"
              rows="2"
              placeholder="Expand what verification should actually look for -- e.g. 'compare specifically against datacenter electricity usage'"
              class="w-full text-xs px-2 py-1.5 rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
            />
            <div class="flex items-center gap-2">
              <button
                @click="saveContext"
                :disabled="savingContext"
                class="px-2 py-1 text-xs rounded-md bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white transition-colors"
              >
                {{ savingContext ? 'Saving...' : 'Save' }}
              </button>
              <button
                @click="contextDraft = localClaim.verification_context || ''; editingContext = false"
                class="text-xs text-gray-400 dark:text-gray-500 hover:underline"
              >
                cancel
              </button>
            </div>
            <p class="text-[10px] text-gray-400 dark:text-gray-500">
              Saving doesn't re-check the claim by itself -- use Force re-verify below to apply it.
            </p>
          </div>
        </div>

        <details v-if="decisions.length" class="mb-3 text-xs">
          <summary class="cursor-pointer text-gray-500 dark:text-gray-400">Automation history ({{ decisions.length }})</summary>
          <div class="mt-1.5 space-y-1.5">
            <div v-for="decision in decisions" :key="decision.id" class="rounded bg-gray-50 dark:bg-gray-900 px-2 py-1.5">
              <p class="font-medium text-gray-700 dark:text-gray-300">{{ decision.decision }}</p>
              <p v-if="decision.reasoning" class="text-gray-500 dark:text-gray-400">{{ decision.reasoning }}</p>
            </div>
          </div>
        </details>

        <div class="flex items-center gap-2 mb-3">
          <button
            @click="runVerify"
            :disabled="verifying"
            class="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50 transition-colors"
          >
            <ShieldCheck class="w-3.5 h-3.5" />
            {{ verifying ? 'Verifying...' : 'Verify' }}
          </button>
          <label class="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400 cursor-pointer select-none">
            <input type="checkbox" v-model="verifyForce" class="accent-blue-600" />
            Force re-verify
          </label>
          <button
            v-if="localClaim.status === 'supported'"
            @click="runCounterEvidence"
            :disabled="findingCounter"
            class="text-xs text-blue-600 dark:text-blue-400 hover:underline disabled:opacity-50"
            title="Find a bounded counter-view for balance; this never changes the verdict"
          >
            {{ findingCounter ? 'Searching...' : 'Find counter-view' }}
          </button>
        </div>

        <template v-if="detail.contradicting_claims?.length">
          <p class="text-xs font-semibold text-red-600 dark:text-red-400 mb-1 flex items-center gap-1">
            <ThumbsDown class="w-3.5 h-3.5" />
            Contradicted by
          </p>
          <div class="space-y-2 mb-3">
            <div
              v-for="rc in detail.contradicting_claims"
              :key="rc.id"
              class="text-xs border-l-2 border-red-300 dark:border-red-700 pl-2"
            >
              <p class="text-gray-700 dark:text-gray-300">{{ rc.canonical_text }}</p>
              <p v-if="rc.reason" class="text-gray-500 dark:text-gray-400 italic">{{ rc.reason }}</p>
              <div class="flex flex-wrap items-center gap-x-2 mt-0.5">
                <a
                  v-for="s in rc.sources"
                  :key="s.source_id"
                  href="#"
                  @click.prevent="router.push(`/sources/${s.source_id}`)"
                  class="text-blue-600 dark:text-blue-400 hover:underline"
                >
                  {{ s.source_title || s.canonical_uri }}
                </a>
                <span
                  class="px-1 py-0.5 rounded uppercase text-[10px] font-medium"
                  :class="rc.candidate_status === 'open'
                    ? 'bg-yellow-100 dark:bg-yellow-900/40 text-yellow-700 dark:text-yellow-300'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500'"
                >
                  {{ rc.candidate_status === 'open' ? 'pending review' : rc.candidate_status }}
                </span>
                <template v-if="rc.candidate_status === 'open'">
                  <button
                    @click="reviewContradiction(rc, 'accepted')"
                    :disabled="reviewingCandidateId === rc.candidate_id"
                    class="text-green-600 dark:text-green-400 hover:underline disabled:opacity-50"
                  >
                    {{ reviewingCandidateId === rc.candidate_id ? 'Working...' : 'Accept' }}
                  </button>
                  <button
                    v-if="reviewingCandidateId !== rc.candidate_id"
                    @click="reviewContradiction(rc, 'rejected')"
                    class="text-red-600 dark:text-red-400 hover:underline"
                  >
                    Reject
                  </button>
                </template>
              </div>
            </div>
          </div>
        </template>

        <template v-if="detail.counter_claims?.length">
          <p class="text-xs font-semibold text-blue-600 dark:text-blue-400 mb-1 flex items-center gap-1">
            <ThumbsDown class="w-3.5 h-3.5" /> Counter-view (for balance)
          </p>
          <div class="space-y-2 mb-3">
            <div v-for="cc in detail.counter_claims" :key="cc.candidate_id" class="text-xs border-l-2 border-blue-300 dark:border-blue-700 pl-2">
              <p class="text-gray-700 dark:text-gray-300">{{ cc.canonical_text }}</p>
              <p v-if="cc.reason" class="text-gray-500 dark:text-gray-400 italic">{{ cc.reason }}</p>
            </div>
          </div>
        </template>

        <template v-if="detail.supporting_claims?.length">
          <p class="text-xs font-semibold text-green-600 dark:text-green-400 mb-1 flex items-center gap-1">
            <ThumbsUp class="w-3.5 h-3.5" />
            Supported by
          </p>
          <div class="space-y-2 mb-3">
            <div
              v-for="sc in detail.supporting_claims"
              :key="sc.id"
              class="text-xs border-l-2 border-green-300 dark:border-green-700 pl-2"
            >
              <p class="text-gray-700 dark:text-gray-300">{{ sc.canonical_text }}</p>
              <div class="flex flex-wrap items-center gap-x-2 mt-0.5">
                <a
                  v-for="s in sc.sources"
                  :key="s.source_id"
                  href="#"
                  @click.prevent="router.push(`/sources/${s.source_id}`)"
                  class="text-blue-600 dark:text-blue-400 hover:underline"
                >
                  {{ s.source_title || s.canonical_uri }}
                </a>
              </div>
            </div>
          </div>
        </template>

        <p
          v-if="['supported', 'contradicted', 'mixed'].includes(localClaim.status)
            && !detail.supporting_claims?.length
            && !detail.contradicting_claims?.length"
          class="text-xs text-gray-400 dark:text-gray-500 mb-3 italic"
        >
          Verified before this detail was tracked -- re-verify (force) to see the specific sources.
        </p>

        <p class="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1">Evidence</p>
        <div class="space-y-2">
          <div
            v-for="ev in detail.evidence"
            :key="ev.id"
            class="flex items-start justify-between gap-2 text-xs border-l-2 border-gray-200 dark:border-gray-600 pl-2"
          >
            <div class="min-w-0">
              <p class="font-medium text-gray-700 dark:text-gray-300 flex items-center gap-1">
                {{ ev.source_title || ev.canonical_uri }}
                <Star
                  v-if="detail.claim.preferred_source_id === ev.source_id"
                  class="w-3 h-3 text-yellow-500 fill-yellow-500"
                />
              </p>
              <p v-if="ev.excerpt_text" class="text-gray-500 dark:text-gray-400 italic">"{{ ev.excerpt_text }}"</p>
            </div>
            <button
              v-if="detail.claim.preferred_source_id !== ev.source_id"
              @click="makePreferred(ev.source_id)"
              class="shrink-0 text-blue-600 dark:text-blue-400 hover:underline whitespace-nowrap"
            >
              Make preferred
            </button>
          </div>
          <p v-if="!detail.evidence?.length" class="text-xs text-gray-400 dark:text-gray-500">No evidence recorded.</p>
        </div>
      </template>
    </div>
  </div>
</template>
