<script setup>
import { Search, History, FolderClock, GitMerge, Database, Quote, ShieldCheck, Sun, Moon } from 'lucide-vue-next'
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const dark = ref(true)

onMounted(() => {
  const saved = localStorage.getItem('darkMode')
  dark.value = saved !== null ? saved === 'true' : true
  applyTheme()
})

function toggleTheme() {
  dark.value = !dark.value
  localStorage.setItem('darkMode', dark.value)
  applyTheme()
}

function applyTheme() {
  document.documentElement.classList.toggle('dark', dark.value)
}
</script>

<template>
  <div class="min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-300">
    <!-- Header -->
    <header class="sticky top-0 z-40 bg-white dark:bg-gray-800 shadow">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
        <div class="flex items-center gap-4">
          <h1
            class="text-xl font-bold text-gray-900 dark:text-white cursor-pointer"
            @click="router.push('/')"
          >
            Deep Research
          </h1>
          <nav class="hidden sm:flex items-center gap-1">
            <router-link
              to="/"
              class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              active-class="bg-gray-100 dark:bg-gray-700"
            >
              <Search class="w-4 h-4" />
              Research
            </router-link>
            <router-link
              to="/history"
              class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              active-class="bg-gray-100 dark:bg-gray-700"
            >
              <History class="w-4 h-4" />
              History
            </router-link>
            <router-link
              to="/sources"
              class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              active-class="bg-gray-100 dark:bg-gray-700"
            >
              <Database class="w-4 h-4" />
              Sources
            </router-link>
            <router-link
              to="/claims"
              class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              active-class="bg-gray-100 dark:bg-gray-700"
            >
              <Quote class="w-4 h-4" />
              Claims
            </router-link>
            <router-link
              to="/topics"
              class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              active-class="bg-gray-100 dark:bg-gray-700"
            >
              <FolderClock class="w-4 h-4" />
              Topics
            </router-link>
            <router-link
              to="/resolution-queue"
              class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              active-class="bg-gray-100 dark:bg-gray-700"
            >
              <GitMerge class="w-4 h-4" />
              Review
            </router-link>
            <router-link
              to="/verification-status"
              class="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              active-class="bg-gray-100 dark:bg-gray-700"
            >
              <ShieldCheck class="w-4 h-4" />
              Verification
            </router-link>
          </nav>
        </div>
        <button
          @click="toggleTheme"
          class="p-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          :title="dark ? 'Switch to light mode' : 'Switch to dark mode'"
        >
          <Moon v-if="dark" class="w-4 h-4" />
          <Sun v-else class="w-4 h-4" />
        </button>
      </div>
    </header>

    <!-- Main content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <router-view />
    </main>
  </div>
</template>
