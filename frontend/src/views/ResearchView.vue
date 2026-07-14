<script setup>
import { ref, onMounted, nextTick, watch, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Send, Loader, Globe, FileSearch, Bot, ChevronDown, Database, Server } from 'lucide-vue-next'
import { marked } from 'marked'
import { useApi } from '../composables/useApi.js'

const route = useRoute()
const router = useRouter()
const api = useApi()

const query = ref('')
const model = ref('')
const models = ref([])
const modelDropdownOpen = ref(false)
const messages = ref([])
const status = ref(null)
const isResearching = ref(false)
const sessionId = ref(null)
const messagesContainer = ref(null)
// Decision 23 (hybrid retrieval): let the user opt into checking the local
// knowledge base first. With this off, research always starts on the web.
// Persisted the same way dark mode is, since it's a standing preference.
const prioritizeKb = ref(localStorage.getItem('prioritizeKb') === 'true')
// Which local model server the agent talks to -- see
// deep_research/model_backends.py. Persisted like prioritizeKb, since it's
// also a standing preference, not a per-query choice.
const backend = ref(localStorage.getItem('llmBackend') || 'ollama')
let abortController = null

watch(prioritizeKb, (val) => {
  localStorage.setItem('prioritizeKb', val)
})

watch(backend, (val) => {
  localStorage.setItem('llmBackend', val)
})

async function loadModelsForBackend() {
  const data = await api.fetchModels(backend.value)
  models.value = data.models || []
  model.value = data.default || (models.value[0] ?? '')
}

function switchBackend(newBackend) {
  if (newBackend === backend.value) return
  backend.value = newBackend
  loadModelsForBackend()
}

onMounted(async () => {
  await loadModelsForBackend()

  // If resuming a session
  if (route.params.id) {
    sessionId.value = route.params.id
    const sessionData = await api.fetchSession(route.params.id)
    if (sessionData.messages) {
      for (const msg of sessionData.messages) {
        if (msg.role === 'user') {
          messages.value.push({ role: 'user', content: msg.content })
        } else if (msg.role === 'assistant' && msg.content) {
          messages.value.push({ role: 'assistant', content: msg.content })
        }
      }
    }
  }
})

watch(messages, () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}, { deep: true })

function renderMarkdown(text) {
  if (!text) return ''
  return marked.parse(text, { breaks: true })
}

function selectModel(m) {
  model.value = m
  modelDropdownOpen.value = false
}

// llama.cpp reports the full gguf file path as the model "id" (e.g.
// /home/.../Qwen_Qwen3-14B-GGUF_Qwen3-14B-Q4_K_M.gguf) -- Ollama's tags
// (gemma3:12b) are already short, so this only shortens the path-like case.
function displayModelName(m) {
  if (!m) return m
  return m.includes('/') ? m.slice(m.lastIndexOf('/') + 1) : m
}

async function submitQuery() {
  const q = query.value.trim()
  if (!q || isResearching.value) return

  messages.value.push({ role: 'user', content: q })
  query.value = ''
  isResearching.value = true
  status.value = { step: 'starting', detail: 'Initializing...' }

  abortController = api.streamResearch(q, model.value, sessionId.value, {
    onSession(data) {
      sessionId.value = data.session_id
    },
    onStatus(data) {
      status.value = data
    },
    onTool(data) {
      status.value = { step: 'tool', detail: `${data.tool}(${Object.values(data.args || {}).join(', ').slice(0, 60)})` }
    },
    onAnswer(data) {
      messages.value.push({ role: 'assistant', content: data.answer })
      sessionId.value = data.session_id
      status.value = null
      isResearching.value = false
    },
    onError(data) {
      messages.value.push({ role: 'assistant', content: `Error: ${data.error}` })
      status.value = null
      isResearching.value = false
    },
    onDone() {
      status.value = null
      isResearching.value = false
    },
  }, prioritizeKb.value, backend.value)
}

function stopResearch() {
  if (abortController) {
    abortController.abort()
    isResearching.value = false
    status.value = null
  }
}

function handleKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    submitQuery()
  }
}

const statusIcon = computed(() => {
  if (!status.value) return null
  switch (status.value.step) {
    case 'scraping': return FileSearch
    case 'tool': return Globe
    default: return Bot
  }
})

const statusText = computed(() => {
  if (!status.value) return ''
  switch (status.value.step) {
    case 'scraping': return `Scraping: ${status.value.detail}`
    case 'searching': return `Searching: ${status.value.detail}`
    case 'generating': return status.value.detail
    case 'thinking': return status.value.detail
    case 'tool': return status.value.detail
    default: return status.value.detail || 'Working...'
  }
})
</script>

<template>
  <div class="flex flex-col h-[calc(100vh-5rem)]">
    <!-- Messages area -->
    <div
      ref="messagesContainer"
      class="flex-1 overflow-y-auto scrollbar-hide space-y-4 pb-4"
    >
      <!-- Empty state -->
      <div
        v-if="messages.length === 0 && !isResearching"
        class="flex flex-col items-center justify-center h-full text-center"
      >
        <Bot class="w-12 h-12 text-gray-400 dark:text-gray-600 mb-4" :stroke-width="1.5" />
        <h2 class="text-xl font-bold text-gray-900 dark:text-white mb-2">Deep Research</h2>
        <p class="text-sm text-gray-500 dark:text-gray-400 max-w-md">
          Ask a question, search the web, or paste a URL to scrape and analyze.
        </p>
      </div>

      <!-- Messages -->
      <div
        v-for="(msg, i) in messages"
        :key="i"
        :class="[
          'rounded-lg px-4 py-3',
          msg.role === 'user'
            ? 'bg-blue-600 text-white ml-12 sm:ml-24'
            : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 mr-4 sm:mr-12'
        ]"
      >
        <div
          v-if="msg.role === 'assistant'"
          class="markdown-content text-sm"
          v-html="renderMarkdown(msg.content)"
        />
        <div v-else class="text-sm whitespace-pre-wrap">{{ msg.content }}</div>
      </div>

      <!-- Status indicator -->
      <div
        v-if="status"
        class="flex items-center gap-2 px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg mr-4 sm:mr-12"
      >
        <component
          :is="statusIcon"
          class="w-4 h-4 text-blue-500 animate-pulse-dot"
          :stroke-width="1.5"
        />
        <span class="text-sm text-gray-500 dark:text-gray-400">{{ statusText }}</span>
      </div>
    </div>

    <!-- Input area -->
    <div class="border-t border-gray-200 dark:border-gray-700 pt-4">
      <!-- Model selector -->
      <div class="flex items-center gap-2 mb-3">
        <div class="flex items-center rounded-md bg-gray-200 dark:bg-gray-700 text-xs overflow-hidden" title="Which local model server the agent talks to">
          <button
            v-for="b in ['ollama', 'llama_cpp']"
            :key="b"
            @click="switchBackend(b)"
            :class="[
              'flex items-center gap-1 px-2.5 py-1.5 transition-colors',
              backend === b
                ? 'bg-blue-600 text-white'
                : 'hover:bg-gray-300 dark:hover:bg-gray-600'
            ]"
          >
            <Server class="w-3.5 h-3.5" :stroke-width="1.5" />
            {{ b === 'ollama' ? 'Ollama' : 'llama.cpp' }}
          </button>
        </div>

        <div class="relative">
          <button
            @click="modelDropdownOpen = !modelDropdownOpen"
            class="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            :title="model"
          >
            <Bot class="w-3.5 h-3.5" :stroke-width="1.5" />
            {{ displayModelName(model) || 'Select model' }}
            <ChevronDown class="w-3 h-3" />
          </button>
          <div
            v-if="modelDropdownOpen"
            class="absolute bottom-full left-0 mb-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg z-50 min-w-48"
          >
            <button
              v-for="m in models"
              :key="m"
              @click="selectModel(m)"
              :title="m"
              :class="[
                'w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors',
                m === model ? 'text-blue-500 font-medium' : ''
              ]"
            >
              {{ displayModelName(m) }}
            </button>
          </div>
        </div>

        <label
          class="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors cursor-pointer select-none"
          title="When enabled, check saved sources before searching the web. Leave it off to start with a web search."
        >
          <input type="checkbox" v-model="prioritizeKb" class="accent-blue-600" />
          <Database class="w-3.5 h-3.5" :stroke-width="1.5" />
          Use saved sources first
        </label>
      </div>

      <!-- Query input -->
      <div class="flex gap-2">
        <textarea
          v-model="query"
          @keydown="handleKeydown"
          :disabled="isResearching"
          placeholder="Ask a question, search the web, or paste a URL to analyze..."
          rows="2"
          class="flex-1 resize-none rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 placeholder-gray-400"
        />
        <button
          v-if="!isResearching"
          @click="submitQuery"
          :disabled="!query.trim()"
          class="px-4 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white transition-colors"
        >
          <Send class="w-4 h-4" :stroke-width="1.5" />
        </button>
        <button
          v-else
          @click="stopResearch"
          class="px-4 rounded-lg bg-red-500 hover:bg-red-600 text-white transition-colors"
          title="Stop research"
        >
          <Loader class="w-4 h-4 animate-spin" :stroke-width="1.5" />
        </button>
      </div>
    </div>
  </div>
</template>
