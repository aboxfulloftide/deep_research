<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { Clock, Trash2, MessageSquare } from 'lucide-vue-next'
import { useApi } from '../composables/useApi.js'

const router = useRouter()
const api = useApi()

const sessions = ref([])
const loading = ref(true)

onMounted(async () => {
  await loadSessions()
})

async function loadSessions() {
  loading.value = true
  const data = await api.fetchSessions()
  sessions.value = data.sessions || []
  loading.value = false
}

async function deleteSession(id, event) {
  event.stopPropagation()
  if (!confirm('Delete this research session?')) return
  await api.deleteSession(id)
  sessions.value = sessions.value.filter(s => s.id !== id)
}

function openSession(id) {
  router.push({ name: 'session', params: { id } })
}

function formatDate(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  return d.toLocaleDateString(undefined, {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}
</script>

<template>
  <div>
    <h2 class="text-xl font-bold text-gray-900 dark:text-white mb-4">Research History</h2>

    <!-- Loading -->
    <div v-if="loading" class="text-sm text-gray-500 dark:text-gray-400">Loading...</div>

    <!-- Empty -->
    <div
      v-else-if="sessions.length === 0"
      class="text-center py-12 text-gray-500 dark:text-gray-400"
    >
      <MessageSquare class="w-10 h-10 mx-auto mb-3 opacity-50" :stroke-width="1.5" />
      <p class="text-sm">No research sessions yet.</p>
    </div>

    <!-- Session list -->
    <div v-else class="space-y-2">
      <div
        v-for="session in sessions"
        :key="session.id"
        @click="openSession(session.id)"
        class="flex items-center justify-between px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors"
      >
        <div class="flex-1 min-w-0">
          <p class="text-sm font-medium text-gray-900 dark:text-white truncate">
            {{ session.title || '(Untitled)' }}
          </p>
          <div class="flex items-center gap-1.5 mt-1">
            <Clock class="w-3 h-3 text-gray-400" :stroke-width="1.5" />
            <span class="text-xs text-gray-500 dark:text-gray-400">
              {{ formatDate(session.updated_at) }}
            </span>
          </div>
        </div>
        <button
          @click="deleteSession(session.id, $event)"
          class="p-2 rounded-md hover:bg-red-50 dark:hover:bg-red-900/30 text-gray-400 hover:text-red-500 transition-colors"
          title="Delete session"
        >
          <Trash2 class="w-4 h-4" :stroke-width="1.5" />
        </button>
      </div>
    </div>
  </div>
</template>
