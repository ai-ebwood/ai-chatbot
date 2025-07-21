<script lang="ts">
  let { children, href, title } = $props();
</script>

<a {href} {title} target="_blank" rel="noopener noreferrer">
  {@render children()}
</a>
