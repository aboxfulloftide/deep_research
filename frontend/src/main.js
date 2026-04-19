import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import './style.css'
import App from './App.vue'
import ResearchView from './views/ResearchView.vue'
import HistoryView from './views/HistoryView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'research', component: ResearchView },
    { path: '/history', name: 'history', component: HistoryView },
    { path: '/session/:id', name: 'session', component: ResearchView },
  ],
})

const app = createApp(App)
app.use(router)
app.mount('#app')
