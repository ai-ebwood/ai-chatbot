<script lang="ts">
  import { autoHeight } from "$lib/actions/autoHeight";
  import ThemeToggle from "$lib/components/ThemeToggle.svelte";
  import Button from "$lib/components/ui/button/button.svelte";
  import Textarea from "$lib/components/ui/textarea/textarea.svelte";
  import { HumanMessage, type BaseMessage, AIMessage } from "$lib/models";
  import axios from "axios";
  import { v4 as uuidv4 } from "uuid";

  let messages: BaseMessage[] = $state([]);
  let inputValue: string = $state("");
  let submitDisabled: boolean = $derived(inputValue.trim() === "");
  const user_id = uuidv4();

  let chatContainer: HTMLDivElement;

  $effect(() => {
    messages;

    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  });

  const handleSubmit = async () => {
    if (submitDisabled) return;
    const question = inputValue.trim();
    inputValue = "";
    messages.push(new HumanMessage(question));

    // loading message
    messages.push(new AIMessage("...", true));

    const result = await axios.post("http://localhost:8000/chat", {
      question: question,
      user_id: user_id,
    });

    console.log(result.data);
    // pop loading messages
    messages.pop();

    messages.push(new AIMessage(result.data.content));
  };
  const handleKeydown = (event: KeyboardEvent) => {
    if (submitDisabled) return;
    if (event.key === "Enter" && event.metaKey) {
      event.preventDefault();
      handleSubmit();
    }
  };
</script>

<div class="flex flex-row p-4 gap-2 h-screen pt-2 w-svw overflow-hidden">
  <div>左边侧边栏</div>
  <div
    class="flex-1 flex flex-col justify-center overflow-y-hidden items-center"
  >
    <div
      bind:this={chatContainer}
      class="flex flex-1 h-full flex-col text-4xl overflow-y-auto w-full max-w-5xl"
    >
      {#each messages as message}
        <div
          class={"flex flex-row my-2 " +
            (message.isHuman() ? "justify-end" : "")}
        >
          <div
            class="text-xl p-4 dark:text-gray-300 rounded-2xl dark:bg-gray-800 w-auto"
          >
            {message.is_loading ? "...":message.content}
          </div>
        </div>
      {/each}
    </div>

    <div class="flex flex-col w-full jusify-center items-center mt-4">
      <form
        class="flex flex-row w-full max-w-5xl gap-4 items-end border border-gray-300 p-4 rounded-4xl"
        onsubmit={handleSubmit}
      >
        <Textarea
          placeholder="Start chatting"
          class="flex-1 resize-none !overflow-y-auto max-h-80 !text-xl !border-none !ring-0 !bg-transparent"
          bind:value={inputValue}
          onkeydown={handleKeydown}
          {@attach autoHeight}
        />
        <Button class="cursor-pointer" type="submit" disabled={submitDisabled}
          ><pre>{"Run ⌘ ⏎"}</pre>
        </Button>
      </form>
    </div>
  </div>
  <div class="hidden md:flex flex-row justify-end">
    <ThemeToggle />
  </div>
</div>
