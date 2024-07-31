import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd())
  const frontendPrefix = env.VITE_PREFIX ? env.VITE_PREFIX : "/"
  console.log(`VITE_PREFIX:${env.VITE_PREFIX}, frontend_prefix:${frontendPrefix}, mode:${mode} ...`)
  return {
    plugins: [vue()],
    base: frontendPrefix
  }
})
