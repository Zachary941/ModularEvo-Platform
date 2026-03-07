import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView,
  },
  {
    path: '/models',
    name: 'models',
    component: () => import('../views/ModelsView.vue'),
  },
  {
    path: '/modularevo',
    name: 'modularevo',
    component: () => import('../views/ModularEvoView.vue'),
  },
  {
    path: '/autorouter',
    name: 'autorouter',
    component: () => import('../views/AutoRouterView.vue'),
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
