<template>
  <div class="h-screen flex justify-center items-center">
    <div class="grid grid-rows-3">
      <h1 class="text-3xl font-extrabold sm:text-5xl">A simple frontend</h1>
      <p class="bg-gray-600 text-white p-4">This is a simple custom frontend made with Vue, tailwindcss and VueForm.</p>
      <HelloWorld msg="Hello world!" />
      </div>
    </div>
</template>

<script setup lang="ts">
import HelloWorld from './components/HelloWorld.vue'
</script>
