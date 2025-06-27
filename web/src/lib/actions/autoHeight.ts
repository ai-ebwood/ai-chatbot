// src/lib/actions/autoHeight.ts

/**
 * A Svelte action that automatically adjusts the height of a textarea to fit its content.
 * It also respects the CSS max-height property.
 */
export function autoHeight(node: HTMLTextAreaElement) {
  const adjustHeight = () => {
    // 关键步骤：
    // 1. 将高度重置为auto，这样scrollHeight才能正确计算出内容所需的完整高度。
    node.style.height = "auto";
    // 2. 将元素的高度设置为其scrollHeight。
    // scrollHeight包含了所有内容（即使是滚动的）所需的高度。
    node.style.height = `${node.scrollHeight}px`;
  };

  // 立即调用一次以设置初始高度
  adjustHeight();

  // 监听输入事件，每次输入都重新计算高度
  node.addEventListener("input", adjustHeight);

  // 当组件销毁时，移除事件监听器，防止内存泄漏
  return () => {
    node.removeEventListener("input", adjustHeight);
  };
}
