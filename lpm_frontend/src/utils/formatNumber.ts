export function formatNumber(stars: number): string {
  return stars >= 1000 ? (stars / 1000).toFixed(1) + 'k' : stars.toString();
}
