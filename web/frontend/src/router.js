import { createRouter, createWebHistory } from 'vue-router'
import HomeView from './views/HomeView.vue'
import Task1View from './views/Task1View.vue'
import Task2View from './views/Task2View.vue'
import DatasetStatsView from './views/DatasetStatsView.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: HomeView
  },
  {
    path: '/task1',
    name: 'Task1',
    component: Task1View
  },
  {
    path: '/task2',
    name: 'Task2',
    component: Task2View
  },
  {
    path: '/stats',
    name: 'DatasetStats',
    component: DatasetStatsView
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
