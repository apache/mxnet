# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

package AI::MXNet::KVStoreServer;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::KVStore;
use Storable;
use MIME::Base64;
use Mouse;
use AI::MXNet::Function::Parameters;

=head1 NAME

    AI::MXNet::KVStoreServer - The key-value store server
=cut

=head2 new

    Initialize a new KVStoreServer.

    Parameters
    ----------
    kvstore : KVStore
=cut

has 'kvstore' => (is => 'ro', isa => 'AI::MXNet::KVStore', required => 1);
has 'handle'  => (is => 'ro', isa => 'KVStoreHandle', default => sub { shift->kvstore->handle }, lazy => 1);
has 'init_logging' => (is => 'rw', isa => 'Int', default => 0);


# return the server controller
method _controller()
{
    return  sub {
        my ($cmd_id, $cmd_body) = @_;
        if (not $self->init_logging)
        {
            ## TODO write logging
            $self->init_logging(1);
        }
        if($cmd_id == 0)
        {
            my $optimizer = Storable::thaw(MIME::Base64::decode_base64($cmd_body));
            $self->kvstore->set_optimizer($optimizer);
        }
        else
        {
            my $rank = $self->kvstore->rank;
            print("server $rank, unknown command ($cmd_id, $cmd_body)\n");
        }
    }
}

=head2 run

    run the server, whose behavior is like
    >>> while receive(x):
    ...     if is_command x: controller(x)
    ...     else if is_key_value x: updater(x)
=cut

method run()
{
    check_call(AI::MXNetCAPI::KVStoreRunServer($self->handle, $self->_controller));
}

# Start server/scheduler
func _init_kvstore_server_module()
{
    my $is_worker = check_call(AI::MXNetCAPI::KVStoreIsWorkerNode());
    if($is_worker == 0)
    {
        my $kvstore = AI::MXNet::KVStore->create('dist');
        my $server = __PACKAGE__->new(kvstore => $kvstore);
        $server->run();
        exit(0);
    }
}

_init_kvstore_server_module();

1;
