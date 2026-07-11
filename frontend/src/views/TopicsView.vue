<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { FolderClock, Plus, MessageSquare } from 'lucide-vue-next'
import { useApi } from '../composables/useApi.js'

const router = useRouter()
const api = useApi()

const topics = ref([])
const loading = ref(true)
const showCreate = ref(false)
const newName = ref('')
const newDescription = ref('')
const creating = ref(false)

const showPasteConversation = ref(false)
const conversationText = ref('')
const conversationTitle = ref('')
const conversationTopicName = ref('')
const pasting = ref(false)
const pasteError = ref(null)

onMounted(loadTopics)

async function loadTopics() {
  loading.value = true
  const data = await api.fetchTopics()
  topics.value = data.topics || []
  loading.value = false
}

async function createTopic() {
  if (!newName.value.trim()) return
  creating.value = true
  try {
    await api.createTopic(newName.value.trim(), newDescription.value.trim())
    newName.value = ''
    newDescription.value = ''
    showCreate.value = false
    await loadTopics()
  } finally {
    creating.value = false
  }
}

async function submitConversation() {
  if (!conversationText.value.trim()) return
  pasting.value = true
  pasteError.value = null
  try {
    const data = await api.ingestConversation(
      conversationText.value.trim(), conversationTitle.value.trim() || null,
      null, conversationTopicName.value.trim() || null,
    )
    // Ingest itself is fast (just saves the text); chunk/extract/verify run
    // in the background on the server -- navigate to the topic right away
    // instead of blocking this page on a multi-minute request. The topic
    // (not a browsable source) is the actual thing to look at, and shows
    // live progress via processingSource so there's no ambiguity about
    // whether leaving this page would interrupt anything (it wouldn't have
    // either way, but now there's nothing slow happening on this page at all).
    conversationText.value = ''
    conversationTitle.value = ''
    conversationTopicName.value = ''
    showPasteConversation.value = false
    router.push({ name: 'topic', params: { id: data.topic_id }, query: { processingSource: data.source_id } })
  } catch (e) {
    pasteError.value = e.message
  } finally {
    pasting.value = false
  }
}

function openTopic(topic) {
  router.push({ name: 'topic', params: { id: topic.slug || topic.id } })
}

function formatDate(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-xl font-bold text-gray-900 dark:text-white">Topics</h2>
      <div class="flex items-center gap-2">
        <button
          @click="showPasteConversation = !showPasteConversation; showCreate = false"
          class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
        >
          <MessageSquare class="w-4 h-4" />
          Paste a Conversation
        </button>
        <button
          @click="showCreate = !showCreate; showPasteConversation = false"
          class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md bg-blue-600 hover:bg-blue-700 text-white transition-colors"
        >
          <Plus class="w-4 h-4" />
          New Topic
        </button>
      </div>
    </div>

    <div
      v-if="showPasteConversation"
      class="mb-6 p-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg space-y-3"
    >
      <input
        v-model="conversationTitle"
        type="text"
        placeholder="Title (optional)"
        class="w-full px-3 py-2 text-sm rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
      />
      <input
        v-model="conversationTopicName"
        type="text"
        placeholder="Topic name (optional -- reuse the same name to group multiple pastes together; defaults to the title)"
        class="w-full px-3 py-2 text-sm rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
      />
      <textarea
        v-model="conversationText"
        rows="8"
        placeholder="Paste a conversation here (e.g. 'User: ...' / 'Assistant: ...' turns). Claims made by any speaker will be extracted and fact-checked."
        class="w-full px-3 py-2 text-sm rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white font-mono"
      ></textarea>
      <button
        @click="submitConversation"
        :disabled="pasting || !conversationText.trim()"
        class="px-3 py-1.5 text-sm rounded-md bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white transition-colors"
      >
        {{ pasting ? 'Ingesting...' : 'Ingest, Extract & Verify' }}
      </button>
      <p v-if="pasteError" class="px-3 py-2 text-sm rounded-md bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300">
        Failed: {{ pasteError }}
      </p>
    </div>

    <div
      v-if="showCreate"
      class="mb-6 p-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg space-y-3"
    >
      <input
        v-model="newName"
        type="text"
        placeholder="Topic name (e.g. AI Investment Bubble)"
        class="w-full px-3 py-2 text-sm rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
      />
      <textarea
        v-model="newDescription"
        placeholder="Description (optional)"
        rows="2"
        class="w-full px-3 py-2 text-sm rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
      />
      <button
        @click="createTopic"
        :disabled="creating || !newName.trim()"
        class="px-3 py-1.5 text-sm rounded-md bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white transition-colors"
      >
        {{ creating ? 'Creating...' : 'Create' }}
      </button>
    </div>

    <div v-if="loading" class="text-sm text-gray-500 dark:text-gray-400">Loading...</div>

    <div
      v-else-if="topics.length === 0"
      class="text-center py-12 text-gray-500 dark:text-gray-400"
    >
      <FolderClock class="w-10 h-10 mx-auto mb-3 opacity-50" :stroke-width="1.5" />
      <p class="text-sm">No topics yet. Create one to start tracking a research area over time.</p>
    </div>

    <div v-else class="space-y-2">
      <div
        v-for="topic in topics"
        :key="topic.id"
        @click="openTopic(topic)"
        class="px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors"
      >
        <p class="text-sm font-medium text-gray-900 dark:text-white flex items-center gap-1.5">
          {{ topic.name }}
          <span
            v-if="topic.topic_type === 'conversation'"
            class="flex items-center gap-1 px-1.5 py-0.5 text-[10px] rounded uppercase font-medium bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300"
          >
            <MessageSquare class="w-2.5 h-2.5" />
            Conversation
          </span>
        </p>
        <p v-if="topic.description" class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
          {{ topic.description }}
        </p>
        <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">Updated {{ formatDate(topic.updated_at) }}</p>
      </div>
    </div>
  </div>
</template>
