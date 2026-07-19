<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { ExternalLink, Radio, RefreshCw } from 'lucide-vue-next'
import { useApi } from '../composables/useApi.js'

const api = useApi()

const providers = ref({})
const recentCalls = ref([])
const loading = ref(true)
const checking = ref(false)
const checkError = ref(null)

let pollHandle = null

onMounted(() => {
  load()
  pollHandle = setInterval(load, 10000)
})

onUnmounted(() => {
  if (pollHandle) clearInterval(pollHandle)
})

async function load() {
  const data = await api.fetchSearchUsage()
  providers.value = data.providers || {}
  recentCalls.value = data.recent_calls || []
  loading.value = false
}

async function checkNow() {
  checking.value = true
  checkError.value = null
  try {
    await api.checkSearchProviders()
    await load()
  } catch (e) {
    checkError.value = e.message
  } finally {
    checking.value = false
  }
}

const PROVIDER_LABELS = {
  duckduckgo: 'DuckDuckGo', bing: 'Bing', mojeek: 'Mojeek', searxng: 'SearXNG (other)',
  wikipedia_api: 'Wikipedia (direct API)', wikidata_api: 'Wikidata (direct API)', brave: 'Brave', brave_fallback: 'Brave (fallback)', tavily: 'Tavily', serper: 'Serper',
}
const PROVIDER_ACCOUNT_URLS = {
  brave: 'https://brave.com/search/api/',
  brave_fallback: 'https://brave.com/search/api/',
  tavily: 'https://app.tavily.com/home#',
  serper: 'https://serper.dev/dashboard',
}
// Preferred order for known providers; anything else (a SearXNG engine we
// haven't explicitly labeled) is appended alphabetically after these.
const KNOWN_ORDER = ['duckduckgo', 'bing', 'mojeek', 'wikipedia_api', 'wikidata_api', 'brave', 'brave_fallback', 'serper', 'tavily']

const providerOrder = computed(() => {
  const keys = Object.keys(providers.value)
  const known = KNOWN_ORDER.filter((k) => keys.includes(k))
  const unknown = keys.filter((k) => !KNOWN_ORDER.includes(k)).sort()
  return [...known, ...unknown]
})

function providerLabel(key) {
  return PROVIDER_LABELS[key] || key
}

function providerAccountUrl(key) {
  return PROVIDER_ACCOUNT_URLS[key]
}

function statusBadge(p) {
  if (!p || !p.last_status) return { label: 'no data', class: 'bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400' }
  if (p.last_status === 'ok') return { label: 'responding', class: 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300' }
  if (p.last_status === 'empty') return { label: 'empty results', class: 'bg-yellow-100 dark:bg-yellow-900/40 text-yellow-700 dark:text-yellow-300' }
  return { label: 'error / rate limited', class: 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300' }
}

function modeBadge(mode) {
  return mode === 'api'
    ? { label: 'api', class: 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300' }
    : { label: 'scrape', class: 'bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300' }
}

function formatDate(iso) {
  if (!iso) return 'never'
  return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit' })
}

const statusColors = {
  ok: 'text-green-600 dark:text-green-400',
  empty: 'text-yellow-600 dark:text-yellow-400',
  error: 'text-red-600 dark:text-red-400',
}
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
        <Radio class="w-5 h-5" />
        Search Provider Usage
      </h2>
      <div class="text-right">
        <button
          @click="checkNow"
          :disabled="checking"
          class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          <RefreshCw class="w-4 h-4" :class="{ 'animate-spin': checking }" />
          {{ checking ? 'Checking...' : 'Check Now' }}
        </button>
        <p v-if="checkError" class="text-xs text-red-500 dark:text-red-400 mt-1">{{ checkError }}</p>
      </div>
    </div>

    <p class="text-sm text-gray-500 dark:text-gray-400 mb-6">
      web_search() combines SearXNG, the direct Wikipedia/Wikidata APIs, Brave, and Serper on each call. Tavily runs
      only when the primary results are thin. Account links are available for providers with usage
      dashboards. "Check Now" fires one live probe at each provider instead of relying on recent calls.
    </p>

    <div v-if="loading" class="text-sm text-gray-500 dark:text-gray-400">Loading...</div>

    <template v-else>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <div
          v-for="key in providerOrder"
          :key="key"
          class="p-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg"
        >
          <div class="flex items-start justify-between gap-2 mb-2">
            <div class="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
              <span class="font-semibold text-gray-900 dark:text-white">{{ providerLabel(key) }}</span>
              <a
                v-if="providerAccountUrl(key)"
                :href="providerAccountUrl(key)"
                target="_blank"
                rel="noopener noreferrer"
                class="inline-flex items-center gap-1 text-xs font-medium text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
                :aria-label="`Open ${providerLabel(key)} account page in a new tab`"
              >
                Account
                <ExternalLink class="w-3 h-3" aria-hidden="true" />
              </a>
            </div>
            <span class="px-1.5 py-0.5 text-[10px] rounded uppercase font-medium" :class="modeBadge(providers[key]?.mode).class">
              {{ modeBadge(providers[key]?.mode).label }}
            </span>
          </div>
          <span class="inline-block px-2 py-0.5 text-xs rounded uppercase font-medium mb-3" :class="statusBadge(providers[key]).class">
            {{ statusBadge(providers[key]).label }}
          </span>
          <div class="text-sm text-gray-700 dark:text-gray-300 space-y-1">
            <p>{{ providers[key]?.calls_today ?? 0 }} call(s) today &middot; {{ providers[key]?.calls_month ?? 0 }} this month</p>
            <p class="text-xs text-gray-500 dark:text-gray-400">
              <span :class="statusColors.ok">ok={{ providers[key]?.ok_count ?? 0 }}</span>
              &middot;
              <span :class="statusColors.empty">empty={{ providers[key]?.empty_count ?? 0 }}</span>
              &middot;
              <span :class="statusColors.error">error={{ providers[key]?.error_count ?? 0 }}</span>
            </p>
            <p class="text-xs text-gray-400 dark:text-gray-500">last call: {{ formatDate(providers[key]?.last_call_at) }}</p>
            <p v-if="providers[key]?.last_error" class="text-xs text-red-500 dark:text-red-400 truncate" :title="providers[key].last_error">
              {{ providers[key].last_error }}
            </p>
          </div>
        </div>
      </div>

      <h3 class="text-sm font-semibold text-gray-900 dark:text-white mb-2">Recent Calls</h3>
      <div v-if="recentCalls.length === 0" class="text-sm text-gray-500 dark:text-gray-400">
        No search calls logged yet.
      </div>
      <div v-else class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-gray-400 dark:text-gray-500 text-xs">
              <th class="pb-1 pr-3">Time</th>
              <th class="pb-1 pr-3">Provider</th>
              <th class="pb-1 pr-3">Mode</th>
              <th class="pb-1 pr-3">Status</th>
              <th class="pb-1 pr-3">Results</th>
              <th class="pb-1 pr-3">Elapsed</th>
              <th class="pb-1">Query</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(c, i) in recentCalls" :key="i" class="border-t border-gray-100 dark:border-gray-800">
              <td class="py-1.5 pr-3 text-gray-500 dark:text-gray-400 whitespace-nowrap">{{ formatDate(c.created_at) }}</td>
              <td class="py-1.5 pr-3">{{ providerLabel(c.provider) }}</td>
              <td class="py-1.5 pr-3">
                <span class="px-1.5 py-0.5 text-[10px] rounded uppercase font-medium" :class="modeBadge(c.mode).class">{{ c.mode }}</span>
              </td>
              <td class="py-1.5 pr-3" :class="statusColors[c.status] || ''">{{ c.status }}</td>
              <td class="py-1.5 pr-3">{{ c.result_count ?? '-' }}</td>
              <td class="py-1.5 pr-3 text-gray-500 dark:text-gray-400 whitespace-nowrap">{{ c.elapsed_ms != null ? `${c.elapsed_ms}ms` : '-' }}</td>
              <td class="py-1.5 truncate max-w-xs text-gray-500 dark:text-gray-400" :title="c.query">{{ c.query }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>
  </div>
</template>
