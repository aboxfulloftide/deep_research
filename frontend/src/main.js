import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import './style.css'
import App from './App.vue'
import ResearchView from './views/ResearchView.vue'
import HistoryView from './views/HistoryView.vue'
import TopicsView from './views/TopicsView.vue'
import TopicDetailView from './views/TopicDetailView.vue'
import ResolutionCandidatesView from './views/ResolutionCandidatesView.vue'
import SourcesView from './views/SourcesView.vue'
import SourceDetailView from './views/SourceDetailView.vue'
import ClaimsView from './views/ClaimsView.vue'
import VerificationStatusView from './views/VerificationStatusView.vue'
import SearchUsageView from './views/SearchUsageView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'research', component: ResearchView },
    { path: '/history', name: 'history', component: HistoryView },
    { path: '/session/:id', name: 'session', component: ResearchView },
    { path: '/topics', name: 'topics', component: TopicsView },
    { path: '/topics/:id', name: 'topic', component: TopicDetailView },
    { path: '/resolution-queue', name: 'resolution-queue', component: ResolutionCandidatesView },
    { path: '/sources', name: 'sources', component: SourcesView },
    { path: '/sources/:id', name: 'source', component: SourceDetailView },
    { path: '/claims', name: 'claims', component: ClaimsView },
    { path: '/verification-status', name: 'verification-status', component: VerificationStatusView },
    { path: '/search-usage', name: 'search-usage', component: SearchUsageView },
  ],
})

const app = createApp(App)
app.use(router)
app.mount('#app')
