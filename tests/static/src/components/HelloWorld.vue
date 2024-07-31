<script setup lang="ts">
import { ref } from 'vue'

defineProps<{ msg: string }>()
const count = ref(0)
</script>

<template>
  <div class="">
    <p>{{ msg }}, count is {{ count }}</p>
    <button @click="count++" class="bg-indigo-400 p-2 font-medium text-white">click me</button>
  </div>
</template>
