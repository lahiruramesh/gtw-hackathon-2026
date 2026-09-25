import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { formatValue } from "@/lib/format";
import type { FieldSpec } from "@/lib/form-fields";

const KIND_LABELS: Record<FieldSpec["kind"], string> = {
  integer: "integer",
  number: "number",
  boolean: "yes / no",
  enum: "choice",
  string: "text",
  multiline: "text",
  secret: "secret",
  checkpoint: "checkpoint",
};

function allowed(field: FieldSpec): string {
  if (field.options) return field.options.map((option) => option.label).join(", ");
  if (field.min === undefined && field.max === undefined) return "—";
  if (field.max === undefined) return `≥ ${formatValue(field.min)}`;
  if (field.min === undefined) return `≤ ${formatValue(field.max)}`;
  return `${formatValue(field.min)} to ${formatValue(field.max)}`;
}

export function ParamsSchemaTable({ fields }: { fields: FieldSpec[] }) {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Parameter</TableHead>
            <TableHead>Type</TableHead>
            <TableHead className="text-right">Default</TableHead>
            <TableHead className="text-right">Allowed</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {fields.map((field) => (
            <TableRow key={field.name}>
              <TableCell className="whitespace-normal">
                <div>{field.label}</div>
                <div className="font-mono text-xs text-muted-foreground">{field.name}</div>
                {field.description && <div className="text-xs text-muted-foreground">{field.description}</div>}
              </TableCell>
              <TableCell className="text-muted-foreground">
                {KIND_LABELS[field.kind]}
                {field.nullable && ", optional"}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">{formatValue(field.defaultValue)}</TableCell>
              <TableCell className="text-right text-xs text-muted-foreground">{allowed(field)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
