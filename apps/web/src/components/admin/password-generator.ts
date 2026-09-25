const ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789";

/** A random temporary password from an unambiguous alphabet (browser crypto). */
export function generatePassword(length = 20): string {
  const bytes = crypto.getRandomValues(new Uint32Array(length));
  return Array.from(bytes, (value) => ALPHABET[value % ALPHABET.length]).join("");
}
