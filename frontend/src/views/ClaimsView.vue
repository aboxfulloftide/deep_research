<script setup>
import { ref, onMounted, computed } from 'vue'
import ClaimListItem from '../components/ClaimListItem.vue'
import { useApi } from '../composables/useApi.js'

const api = useApi()

const claims = ref([])
const loading = ref(true)
const filterText = ref('')

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
      <ClaimListItem v-for="claim in filteredClaims" :key="claim.id" :claim="claim" />
    </div>
  </div>
</template>
