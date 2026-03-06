import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView,
  },
  {
    path: '/chapter3',
    name: 'chapter3',
    component: () => import('../views/Chapter3View.vue'),
  },
  {
    path: '/chapter4',
    name: 'chapter4',
    component: () => import('../views/Chapter4View.vue'),
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
