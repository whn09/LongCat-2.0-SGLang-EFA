output "instance_ids" {
  value = aws_instance.p5en[*].id
}

output "private_ips" {
  value = aws_instance.p5en[*].private_ip
}

output "security_group_id" {
  value = aws_security_group.node.id
}

output "launch_template_id" {
  value = aws_launch_template.p5en.id
}

output "placement_group" {
  value = aws_placement_group.cluster.name
}
