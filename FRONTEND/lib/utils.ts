export function cn(...inputs: Array<string | boolean | null | undefined>) {
  return inputs.filter(Boolean).join(" ")
}
