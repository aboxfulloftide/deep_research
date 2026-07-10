<script setup>
import { ref, onMounted, computed } from 'vue'
import { ShieldCheck, ChevronDown, ChevronRight, Star, ThumbsUp, ThumbsDown } from 'lucide-vue-next'
import { useRouter } from 'vue-router'
import { useApi } from '../composables/useApi.js'

const api = useApi()
const router = useRouter()

const claims = ref([])
const loading = ref(true)
const filterText = ref('')
const expandedId = ref(null)
const detailByClaim = ref({})
const verifyingId = ref(null)
const verifyForce = ref(false)
const reviewingCandidateId = ref(null)

onMounted(load)

async function load() {
  loading.value = true
  const data = await api.fetchClaims(300)
  claims.value = data.claims || []
  loading.value = false
}

const filteredClaims = computed(() => {
  const q = filterText.value.trim().toLowerCase()
  if (!q) return claims.value
  return claims.value.filter(c => c.canonical_text.toLowerCase().includes(q))
})

const statusColors = {
  unverified: 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  supported: 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300',
  contradicted: 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300',
  mixed: 'bg-yellow-100 dark:bg-yellow-900/40 text-yellow-700 dark:text-yellow-300',
  deprecated: 'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 line-through',
}

async function toggleExpand(claim) {
  if (expandedId.value === claim.id) {
    expandedId.value = null
    return
  }
  expandedId.value = claim.id
  if (!detailByClaim.value[claim.id]) {
    const data = await api.fetchClaim(claim.id)
    detailByClaim.value = { ...detailByClaim.value, [claim.id]: data }
  }
}

async function runVerify(claim) {
  verifyingId.value = claim.id
  try {
    const data = await api.verifyClaim(claim.id, verifyForce.value)
    const idx = claims.value.findIndex(c => c.id === claim.id)
    if (idx !== -1) claims.value[idx] = { ...claims.value[idx], status: data.result.status }
    delete detailByClaim.value[claim.id]
    const fresh = await api.fetchClaim(claim.id)
    detailByClaim.value = { ...detailByClaim.value, [claim.id]: fresh }
  } finally {
    verifyingId.value = null
  }
}

async function makePreferred(claim, sourceId) {
  await api.setPreferredSource(claim.id, sourceId)
  const fresh = await api.fetchClaim(claim.id)
  detailByClaim.value = { ...detailByClaim.value, [claim.id]: fresh }
}

async function reviewContradiction(claim, rc, decision) {
  reviewingCandidateId.value = rc.candidate_id
  try {
    await api.reviewResolutionCandidate(rc.candidate_id, decision)
    const fresh = await api.fetchClaim(claim.id)
    detailByClaim.value = { ...detailByClaim.value, [claim.id]: fresh }
    const idx = claims.value.findIndex(c => c.id === claim.id)
    if (idx !== -1) claims.value[idx] = { ...claims.value[idx], status: fresh.claim.status }
  } finally {
    reviewingCandidateId.value = null
  }
}
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-xl font-bold text-gray-900 dark:text-white">Claims</h2>
      <input
        v-model="filterText"
        type="text"
        placeholder="Filter by text..."
        class="px-3 py-1.5 text-sm rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white w-64"
      />
    </div>

    <div v-if="loading" class="text-sm text-gray-500 dark:text-gray-400">Loading...</div>

    <div v-else-if="filteredClaims.length === 0" class="text-center py-12 text-gray-500 dark:text-gray-400">
      <p class="text-sm">No claims match.</p>
    </div>

    <div v-else class="space-y-2">
      <div
        v-for="claim in filteredClaims"
        :key="claim.id"
        class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden"
      >
        <div
          @click="toggleExpand(claim)"
          class="flex items-start gap-2 px-4 py-3 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
        >
          <component :is="expandedId === claim.id ? ChevronDown : ChevronRight" class="w-4 h-4 mt-0.5 shrink-0 text-gray-400" />
          <div class="min-w-0 flex-1">
            <p class="text-sm text-gray-900 dark:text-white">{{ claim.canonical_text }}</p>
            <div class="flex items-center gap-2 mt-1">
              <span class="px-1.5 py-0.5 text-[10px] rounded uppercase font-medium" :class="statusColors[claim.status] || statusColors.unverified">
                {{ claim.status }}
              </span>
              <span class="text-xs text-gray-400 dark:text-gray-500">{{ claim.claim_type }}</span>
              <span v-if="claim.importance_score != null" class="text-xs text-gray-400 dark:text-gray-500">
                importance={{ claim.importance_score.toFixed(2) }}
              </span>
            </div>
          </div>
        </div>

        <div v-if="expandedId === claim.id" class="px-4 pb-3 border-t border-gray-100 dark:border-gray-700 pt-3">
          <div v-if="!detailByClaim[claim.id]" class="text-xs text-gray-400 dark:text-gray-500">Loading evidence...</div>
          <template v-else>
            <div class="flex items-center gap-2 mb-3">
              <button
                @click="runVerify(claim)"
                :disabled="verifyingId === claim.id"
                class="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50 transition-colors"
              >
                <ShieldCheck class="w-3.5 h-3.5" />
                {{ verifyingId === claim.id ? 'Verifying...' : 'Verify' }}
              </button>
              <label class="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400 cursor-pointer select-none">
                <input type="checkbox" v-model="verifyForce" class="accent-blue-600" />
                Force re-verify
              </label>
            </div>

            <template v-if="detailByClaim[claim.id].contradicting_claims?.length">
              <p class="text-xs font-semibold text-red-600 dark:text-red-400 mb-1 flex items-center gap-1">
                <ThumbsDown class="w-3.5 h-3.5" />
                Contradicted by
              </p>
              <div class="space-y-2 mb-3">
                <div
                  v-for="rc in detailByClaim[claim.id].contradicting_claims"
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
                        @click="reviewContradiction(claim, rc, 'accepted')"
                        :disabled="reviewingCandidateId === rc.candidate_id"
                        class="text-green-600 dark:text-green-400 hover:underline disabled:opacity-50"
                      >
                        {{ reviewingCandidateId === rc.candidate_id ? 'Working...' : 'Accept' }}
                      </button>
                      <button
                        v-if="reviewingCandidateId !== rc.candidate_id"
                        @click="reviewContradiction(claim, rc, 'rejected')"
                        class="text-red-600 dark:text-red-400 hover:underline"
                      >
                        Reject
                      </button>
                    </template>
                  </div>
                </div>
              </div>
            </template>

            <template v-if="detailByClaim[claim.id].supporting_claims?.length">
              <p class="text-xs font-semibold text-green-600 dark:text-green-400 mb-1 flex items-center gap-1">
                <ThumbsUp class="w-3.5 h-3.5" />
                Supported by
              </p>
              <div class="space-y-2 mb-3">
                <div
                  v-for="sc in detailByClaim[claim.id].supporting_claims"
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
              v-if="['supported', 'contradicted', 'mixed'].includes(claim.status)
                && !detailByClaim[claim.id].supporting_claims?.length
                && !detailByClaim[claim.id].contradicting_claims?.length"
              class="text-xs text-gray-400 dark:text-gray-500 mb-3 italic"
            >
              Verified before this detail was tracked -- re-verify (force) to see the specific sources.
            </p>

            <p class="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1">Evidence</p>
            <div class="space-y-2">
              <div
                v-for="ev in detailByClaim[claim.id].evidence"
                :key="ev.id"
                class="flex items-start justify-between gap-2 text-xs border-l-2 border-gray-200 dark:border-gray-600 pl-2"
              >
                <div class="min-w-0">
                  <p class="font-medium text-gray-700 dark:text-gray-300 flex items-center gap-1">
                    {{ ev.source_title || ev.canonical_uri }}
                    <Star
                      v-if="detailByClaim[claim.id].claim.preferred_source_id === ev.source_id"
                      class="w-3 h-3 text-yellow-500 fill-yellow-500"
                    />
                  </p>
                  <p v-if="ev.excerpt_text" class="text-gray-500 dark:text-gray-400 italic">"{{ ev.excerpt_text }}"</p>
                </div>
                <button
                  v-if="detailByClaim[claim.id].claim.preferred_source_id !== ev.source_id"
                  @click="makePreferred(claim, ev.source_id)"
                  class="shrink-0 text-blue-600 dark:text-blue-400 hover:underline whitespace-nowrap"
                >
                  Make preferred
                </button>
              </div>
              <p v-if="!detailByClaim[claim.id].evidence?.length" class="text-xs text-gray-400 dark:text-gray-500">No evidence recorded.</p>
            </div>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>
