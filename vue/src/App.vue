<template>
  <div id="app">
    <grabcut msg="GrabCut"></grabcut>
  </div>
</template>

<script>
import grabcut from './components/grabcut.vue'

export default {
  name: 'App',
  components: {
    // HelloWorld,
    grabcut
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}
</style>
